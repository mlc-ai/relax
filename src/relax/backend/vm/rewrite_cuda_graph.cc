/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

class OutputCollector : public ExprVisitor {
 public:
  static std::unordered_set<const VarNode*> Collect(const Function& func) {
    OutputCollector collector;
    collector.VisitExpr(func->body);
    return collector.outputs_;
  }

  void VisitExpr_(const VarNode* var) final {
    if (is_visiting_output_) {
      outputs_.insert(var);
    }
  }

  void VisitExpr_(const SeqExprNode* seq) final {
    bool is_visiting_output = false;
    std::swap(is_visiting_output, is_visiting_output_);
    for (const auto& binding_block : seq->blocks) {
      VisitBindingBlock(binding_block);
    }
    std::swap(is_visiting_output, is_visiting_output_);
    VisitExpr(seq->body);
  }

 private:
  std::unordered_set<const VarNode*> outputs_;
  bool is_visiting_output_ = true;
};

struct StaticRegion {
  Function alloc_func;
  Function capture_func;
  std::vector<std::pair<const VarNode*, int>> alloc_tensor_to_index;
  const VarBindingNode* launch_point = nullptr;
  std::unordered_map<const VarNode*, const VarBindingNode*> bindings;
};

class StaticRegionExtractor : public ExprVisitor {
 public:
  static std::unordered_map<const BindingBlockNode*, std::vector<StaticRegion>> Extract(
      const IRModule& mod, const Function& func) {
    StaticRegionExtractor extractor(mod, OutputCollector::Collect(func));
    extractor.VisitExpr(func->body);
    return extractor.block_to_static_regions_;
  }
  StaticRegionExtractor(const IRModule& mod, const std::unordered_set<const VarNode*>& outputs)
      : mod_(mod), outputs_(outputs) {}

  struct DependencyGraph {
    void AddBinding(const VarBindingNode* binding, bool is_alloc_tensor) {
      const auto* var = binding->var.get();
      if (is_alloc_tensor) {
        alloc_tensors.push_back(var);
      } else {
        nodes.push_back(var);
      }
      bindings.emplace(var, binding);
    }

    void AddEdge(const VarNode* src, const VarNode* dst) {
      dep_src2dst[src].push_back(dst);
      dep_dst2src[dst].push_back(src);
    }

    std::unordered_map<const VarNode*, std::vector<const VarNode*>> dep_src2dst;
    std::unordered_map<const VarNode*, std::vector<const VarNode*>> dep_dst2src;
    std::vector<const VarNode*> alloc_tensors;
    std::vector<const VarNode*> nodes;
    std::unordered_map<const VarNode*, const VarBindingNode*> bindings;
  };

  struct ScopeInfo {
    DependencyGraph graph;
    std::vector<StaticRegion> static_regions;
  };

  void VisitBindingBlock_(const BindingBlockNode* block) final {
    ScopeInfo scope;
    std::swap(scope, scope_);
    for (const auto& binding : block->bindings) {
      VisitBinding(binding);
    }
    SummarizeStaticRegion();
    std::swap(scope, scope_);
    if (scope.static_regions.size()) {
      block_to_static_regions_.emplace(block, std::move(scope.static_regions));
    }
  }

  void VisitBinding_(const MatchCastNode* binding) final {
    ExprVisitor::VisitBinding_(binding);
    SummarizeStaticRegion();
  }

  bool IsStaticAlloc(const CallNode* alloc_tensor_call) {
    auto shape = Downcast<ShapeExpr>(alloc_tensor_call->args[0]);
    return std::all_of(shape->values.begin(), shape->values.end(),
                       [](const PrimExpr& expr) { return expr.as<IntImmNode>() != nullptr; });
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    ExprVisitor::VisitBinding_(binding);

    const Expr& value = binding->value;
    static const auto& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (value->IsInstance<VarNode>()) {
      // var -> var re-binding is not needed as we expect the binding is canonicalized
      return;
    }
    if (value->IsInstance<ConstantNode>()) {
      scope_.graph.AddBinding(binding, false);
      return;
    }
    if (const auto* call = value.as<CallNode>()) {
      if (call->op == alloc_tensor_op && IsStaticAlloc(call)) {
        scope_.graph.AddBinding(binding, true);
        return;
      } else if (const auto* gv = call->op.as<GlobalVarNode>()) {
        auto func = mod_->Lookup(GetRef<GlobalVar>(gv));
        if (!func->IsInstance<tir::PrimFuncNode>()) {
          return;
        }
        bool is_all_args_static =
            std::all_of(call->args.begin(), call->args.end(), [&](const Expr& arg) -> bool {
              if (const auto* var = arg.as<VarNode>()) {
                return scope_.graph.bindings.count(var);
              } else if (arg->IsInstance<IntImmNode>() || arg->IsInstance<FloatImmNode>() ||
                         arg->IsInstance<ConstantNode>()) {
                return true;
              }
              return false;
            });
        if (is_all_args_static) {
          scope_.graph.AddBinding(binding, false);
          for (const Expr& arg : call->args) {
            if (const auto* var = arg.as<VarNode>()) {
              scope_.graph.AddEdge(var, binding->var.get());
            }
          }
          return;
        }
      }
    }
    SummarizeStaticRegion();
  }

  void PruneOutputNodes(DependencyGraph* graph, const std::unordered_set<const VarNode*> outputs) {
    std::unordered_set<const VarNode*> src_nodes;
    std::unordered_set<const VarNode*> visited;

    // backward pass to find all alloc_tensor for outputs
    std::vector<const VarNode*> task_stack{outputs.begin(), outputs.end()};
    auto f_swim = [&](const VarNode* node) {
      if (visited.count(node)) return;
      visited.insert(node);
      auto it = graph->dep_dst2src.find(node);
      if (it == graph->dep_dst2src.end()) {
        src_nodes.insert(node);
        return;
      }
      for (const auto* src : it->second) {
        task_stack.push_back(src);
      }
    };
    while (!task_stack.empty()) {
      const VarNode* node = task_stack.back();
      task_stack.pop_back();
      f_swim(node);
    }
    // forward pass to prune all intermediate bindings involved with output tensors
    task_stack = {src_nodes.begin(), src_nodes.end()};
    visited.clear();
    std::unordered_set<const VarNode*> nodes_to_remove{src_nodes.begin(), src_nodes.end()};
    auto f_sink = [&](const VarNode* node) {
      if (visited.count(node)) return;
      visited.insert(node);
      auto it = graph->dep_src2dst.find(node);
      if (it == graph->dep_src2dst.end()) return;
      for (const auto* dst : it->second) {
        nodes_to_remove.insert(dst);
        task_stack.push_back(dst);
      }
    };
    while (!task_stack.empty()) {
      const VarNode* node = task_stack.back();
      task_stack.pop_back();
      f_sink(node);
    }
    // remove nodes from the graph
    for (const auto* node : nodes_to_remove) {
      graph->bindings.erase(node);
    }
    auto f_is_removed = [&](const VarNode* node) { return nodes_to_remove.count(node); };
    graph->nodes.erase(std::remove_if(graph->nodes.begin(), graph->nodes.end(), f_is_removed),
                       graph->nodes.end());
    graph->alloc_tensors.erase(
        std::remove_if(graph->alloc_tensors.begin(), graph->alloc_tensors.end(), f_is_removed),
        graph->alloc_tensors.end());
  }

  Function EmitStaticAllocations(BlockBuilderNode* builder, StaticRegion* region) {
    builder->BeginBindingBlock();
    Array<Expr> alloc_tensors;
    Array<StructInfo> alloc_tensors_struct_info;
    for (const auto& var : scope_.graph.alloc_tensors) {
      region->alloc_tensor_to_index.emplace_back(var, alloc_tensors.size());
      alloc_tensors.push_back(builder->Emit(scope_.graph.bindings.at(var)->value));
      alloc_tensors_struct_info.push_back(
          Downcast<StructInfo>(GetRef<Var>(var)->struct_info_.value()));
    }
    auto output = builder->Emit(Tuple(alloc_tensors));
    auto block = builder->EndBlock();
    auto func_body = builder->Normalize(SeqExpr({block}, output));
    auto func = Function({}, func_body, TupleStructInfo(alloc_tensors_struct_info));
    return func;
  }

  Function EmitStaticGraph(BlockBuilderNode* builder, StaticRegion* region) {
    Var allocs{"allocs", region->alloc_func->ret_struct_info};
    builder->BeginBindingBlock();
    for (int i = 0; i < static_cast<int>(scope_.graph.alloc_tensors.size()); ++i) {
      const VarNode* tensor = scope_.graph.alloc_tensors[i];
      const Expr alloc = builder->Normalize(TupleGetItem(allocs, i));
      builder->EmitNormalized(VarBinding(GetRef<Var>(tensor), alloc));
    }
    for (const VarNode* node : scope_.graph.nodes) {
      const VarBindingNode* binding = scope_.graph.bindings.at(node);
      builder->EmitNormalized(GetRef<VarBinding>(binding));
    }
    auto void_output = builder->Emit(Tuple(Array<Expr>{}));
    auto block = builder->EndBlock();
    auto func_body = builder->Normalize(SeqExpr({block}, void_output));
    auto func = Function({allocs}, func_body, TupleStructInfo(Array<StructInfo>{}));
    return func;
  }

  void SummarizeStaticRegion() {
    PruneOutputNodes(&scope_.graph, outputs_);

    // Find all bindings other than alloc_tensors to emit
    std::vector<VarBinding> bindings_to_emit;
    bool has_kernel_launch = false;
    for (const auto* var : scope_.graph.nodes) {
      auto it = scope_.graph.bindings.find(var);
      const VarBindingNode* binding = it->second;
      if (const CallNode* call = binding->value.as<CallNode>();
          call != nullptr && call->op.as<GlobalVarNode>()) {
        has_kernel_launch = true;
      }
      bindings_to_emit.push_back(GetRef<VarBinding>(binding));
    }
    if (!has_kernel_launch) return;

    BlockBuilder static_region_builder = BlockBuilder::Create(mod_);

    StaticRegion region;
    // Emit alloc_tensors as a separate function
    region.alloc_func = EmitStaticAllocations(static_region_builder.operator->(), &region);
    // Emit the rest of the graph, with original alloc_tensors replaced with the function parameters
    region.capture_func = EmitStaticGraph(static_region_builder.operator->(), &region);
    region.launch_point = scope_.graph.bindings.at(scope_.graph.nodes.front());
    region.bindings = std::move(scope_.graph.bindings);

    scope_.static_regions.push_back(std::move(region));
    scope_.graph = DependencyGraph();
  }

 private:
  IRModule mod_;
  std::unordered_set<const VarNode*> outputs_;
  ScopeInfo scope_;
  std::unordered_map<const BindingBlockNode*, std::vector<StaticRegion>> block_to_static_regions_;
};

class CUDAGraphRewriter : public ExprMutator {
 public:
  explicit CUDAGraphRewriter(IRModule mod) : ExprMutator(mod) {}

  IRModule Rewrite() {
    auto mod = builder_->GetContextIRModule();
    for (const auto& [gv, func] : mod->functions) {
      if (const auto* func_node = func.as<FunctionNode>()) {
        binding_block_to_regions_ =
            StaticRegionExtractor::Extract(mod, GetRef<Function>(func_node));
        Function new_func = Downcast<Function>(func);
        auto fptr = new_func.CopyOnWrite();
        fptr->body = VisitExpr(std::move(fptr->body));
        builder_->UpdateFunction(gv, new_func);
      }
    }
    return builder_->GetContextIRModule();
  }

  Expr MakeGetCapturedCUDAGraph(const GlobalVar& gv_alloc, const GlobalVar& gv_capture,
                                const StructInfo& alloc_tensors_sinfo) {
    return Call(call_builtin_with_ctx_op_,
                {builtin_get_captured_cuda_graph_, Tuple({gv_alloc, gv_capture})}, Attrs(),
                {TupleStructInfo({ObjectStructInfo(), alloc_tensors_sinfo})});
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* binding_block) final {
    builder_->BeginBindingBlock();
    if (auto it = binding_block_to_regions_.find(binding_block);
        it != binding_block_to_regions_.end()) {
      for (const auto& region : it->second) {
        auto gv_alloc = builder_->AddFunction(region.alloc_func, "cuda_graph_capture_func_alloc");
        auto gv_capture =
            builder_->AddFunction(region.capture_func, "cuda_graph_capture_func_capture");
        auto graph_tensors = builder_->Emit(
            MakeGetCapturedCUDAGraph(gv_alloc, gv_capture, region.alloc_func->ret_struct_info));

        for (const auto& binding : region.bindings) {
          bindings_to_remove_.insert(binding.first);
        }
        auto tensors = builder_->Emit(TupleGetItem(graph_tensors, 1));
        for (const auto& [var, index] : region.alloc_tensor_to_index) {
          // TupleGetItem is emitted lazily because not all the tensors are needed by the rest of
          // the program
          auto tensor = TupleGetItem(tensors, index);
          var_remap_[var] = {nullptr, std::move(tensor)};
        }
        auto graph = builder_->Emit(TupleGetItem(graph_tensors, 0));
        graph_launch_point_[region.launch_point->var.get()] = graph.get();
      }
    }
    for (const auto& binding : binding_block->bindings) {
      VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    if (auto it = graph_launch_point_.find(binding->var.get()); it != graph_launch_point_.end()) {
      // Emit launch graph
      builder_->Emit(Call(builtin_cuda_graph_launch_, {GetRef<Var>(it->second)}, Attrs(),
                          {TupleStructInfo(Array<StructInfo>{})}));
      return;
    } else if (bindings_to_remove_.count(binding->var.get())) {
      return;
    }
    ExprMutator::VisitBinding_(binding);
  }

  Expr VisitExpr_(const VarNode* var) final {
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      auto& [var, expr] = it->second;
      if (var == nullptr) {
        var = builder_->Emit(expr).get();
      }
      return GetRef<Var>(var);
    }
    return GetRef<Expr>(var);
  }

 private:
  std::unordered_set<const VarNode*> bindings_to_remove_;
  std::unordered_map<const VarNode*, std::pair<const VarNode*, Expr>> var_remap_;
  std::unordered_map<const VarNode*, const VarNode*> graph_launch_point_;
  std::unordered_map<const BindingBlockNode*, std::vector<StaticRegion>> binding_block_to_regions_;
  const Op& call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const ExternFunc builtin_get_captured_cuda_graph_{"vm.builtin.get_captured_cuda_graph"};
  const ExternFunc builtin_cuda_graph_launch_{"vm.builtin.cuda_graph_launch"};
};

IRModule RewriteCUDAGraph(IRModule mod) {
  CUDAGraphRewriter rewriter(mod);
  mod = rewriter.Rewrite();
  return mod;
}

namespace transform {

Pass RewriteCUDAGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return ::tvm::relax::RewriteCUDAGraph(std::move(m)); };
  return CreateModulePass(pass_func, 0, "RewriteCUDAGraph", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RewriteCUDAGraph").set_body_typed(RewriteCUDAGraph);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
