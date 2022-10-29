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
/*!
 * \file src/relax/transform/to_non_dataflow.cc
 * \brief Transform all dataflow structure to non-dataflow version.
 */
#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
namespace tvm {

namespace tir {

PrimExpr ReLU(PrimExpr x) { return max(x, 0); }

PrimExpr BiasAdd(PrimExpr x, PrimExpr bias) { return x + bias; }


// Analyze the function and match it with supported cutlass kernels
struct FuncMatcher : public StmtExprVisitor {
  inline void FirstFuncEnd() {
    is_matching_ = false;
    allocs1_.erase(intermediate_buffer_);
  }

  inline void AppendCutlass(std::string func) {
    std::string kernel_after_append = cutlass_annotation_ + func;
    if (!supported_cutlass_kernel_.count(kernel_after_append)) {
      FirstFuncEnd();
    } else {
      cutlass_annotation_ = kernel_after_append;
    }
  }

  void VisitStmt_(const BlockNode* op) final {
    if (fail_) {
      return;
    }
    if (op->name_hint == "root") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    if (is_matching_) {
      if (block_partition_.empty()) {
        const EvaluateNode* eval = op->body.as<EvaluateNode>();
        if (!eval) {
          fail_ = true;
          return;
        }
        const CallNode* call = eval->value.as<CallNode>();
        if (!call || !call->op.same_as(builtin::cutlass_gemm())) {
          fail_ = true;
          return;
        }
        AppendCutlass("cutlass.dense");
      } else {
        Array<BufferRegion> reads = op->reads;
        Array<BufferRegion> writes = op->writes;
        Array<BufferLoad> single_loads;
        for (const auto& read : op->reads) {
          Array<PrimExpr> indices;
          for (const auto& range : read->region) {
            ICHECK(is_one(range->extent)) << op->name_hint;
            indices.push_back(range->min);
          }
          single_loads.push_back(BufferLoad(read->buffer, indices));
        }
        const auto* buffer_store = op->body.as<BufferStoreNode>();
        if (!buffer_store) {
          FirstFuncEnd();
        }
        PrimExpr rhs = op->body.as<BufferStoreNode>()->value;
        if (single_loads.size() == 1) {
          PrimExpr composite = ReLU(single_loads[0]);
          if (StructuralEqual()(composite, rhs)) {
            AppendCutlass("_relu");
          } else {
            FirstFuncEnd();
          }
        } else if (single_loads.size() == 2) {
          PrimExpr composite = BiasAdd(single_loads[0], single_loads[1]);
          if (StructuralEqual()(composite, rhs)) {
            AppendCutlass("_bias");
          } else {
            FirstFuncEnd();
          }
        } else {
          FirstFuncEnd();
        }
      }
    }
    for (const auto& read : op->reads) {
      if (is_matching_) {
        input1_.insert(read->buffer);
      } else {
        input2_.insert(read->buffer);
      }
    }
    for (const auto& write : op->writes) {
      if (is_matching_) {
        allocs1_.insert(write->buffer);
      } else if (allocs1_.count(write->buffer)) {
        fail_ = true;
        return;
      } else {
        allocs2_.insert(write->buffer);
      }
      if (is_matching_) {
        intermediate_buffer_ = write->buffer;
      } else {
        input2_.insert(write->buffer);
      }
    }
    block_partition_.Set(GetRef<Block>(op), Bool(is_matching_));
    StmtExprVisitor::VisitStmt_(op);
  }
  // alloc_buffers for the first function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs1_;
  // alloc_buffers for the second function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs2_;
  // whether the current block is in the first function
  Map<Block, Bool> block_partition_;
  // input buffers for the first function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input1_;
  // input buffers for the second function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input2_;
  // The output buffer for the first function, which is also the input buffer for the second function
  Buffer intermediate_buffer_;
  // Indicate whether we are in the first function
  bool is_matching_ = true;
  // Indicate whether we have failed. If failed, we will not do any further analysis and directly return the original one.
  bool fail_ = false;
  // The accumulated annotation for which cutlass kernel we are matching.
  std::string cutlass_annotation_;
  // The supported cutlass kernels
  const std::unordered_set<std::string> supported_cutlass_kernel_ = {
      "cutlass.dense", "cutlass.dense_bias", "cutlass.dense_bias_relu"};
};

// mask the removed part with evaluate(0), and update the alloc_buffers for blocks
class BlockMasker : public StmtExprMutator {
 public:
  static Stmt Mask(Stmt stmt, const Map<Block, Bool>& block_partition,
                   const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& allocs,
                   bool is_cutlass) {
    BlockMasker masker(block_partition, allocs, is_cutlass);
    return masker(stmt);
  }

 private:
  BlockMasker(const Map<Block, Bool>& block_partition,
              const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& allocs,
              bool is_cutlass)
      : block_partition_(block_partition), allocs_(allocs), is_cutlass_(is_cutlass) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*block.operator->());
    if (op->name_hint != "root") {
      ICHECK(block_partition_.count(GetRef<Block>(op)));
      bool block_is_cutlass = block_partition_[GetRef<Block>(op)]->value;
      if (!(is_cutlass_ ^ block_is_cutlass)) {
        n->body = block->body;
      } else {
        n->body = Evaluate(0);
        n->reads.clear();
        n->writes.clear();
      }
    }
    Array<Buffer> alloc_buffers;
    for (const Buffer& b : block->alloc_buffers) {
      if (allocs_.count(b)) {
        alloc_buffers.push_back(b);
      }
    }
    n->alloc_buffers = alloc_buffers;
    return Block(n);
  }
  Map<Block, Bool> block_partition_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs_;
  bool is_cutlass_ = false;
};

/*!
  * \brief Split the input function into two functions, one for the cutlass kernel and one for the
  * rest.
  * \param func The input function.
  * \param arg_partition The input arg for the functions after split.
  * \return A pair of functions, the first one is the cutlass kernel and the second one is the rest.
  */
std::pair<PrimFunc, Optional<PrimFunc>> SplitFunctions(
    PrimFunc func, std::vector<std::vector<int>>* arg_partition) {
  FuncMatcher matcher;
  matcher(func->body);
  if (matcher.fail_) {
    return {func, NullOpt};
  }
  bool has_second_func = false;
  for (const auto& pr : matcher.block_partition_) {
    if (!pr.second->value) {
      has_second_func = true;
      break;
    }
  }
  if (!has_second_func) {
    return {WithAttr(func, "cutlass_kernel", String(matcher.cutlass_annotation_)), NullOpt};
  }

  Stmt body1 = BlockMasker::Mask(func->body, matcher.block_partition_, matcher.allocs1_, true);
  Stmt body2 = BlockMasker::Mask(func->body, matcher.block_partition_, matcher.allocs2_, false);
  Array<Var> new_params1;
  std::vector<int> arg_partition1;
  for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
    Var param = func->params[i];
    if (matcher.input1_.count(func->buffer_map[param])) {
      new_params1.push_back(param);
      if (i != static_cast<int>(func->params.size()) - 1) {
        arg_partition1.push_back(i);
      }
    }
  }
  arg_partition->push_back(arg_partition1);

  new_params1.push_back(Var("output", DataType::Handle()));
  Map<Var, Buffer> new_buffer_map1;
  for (const auto& kv : func->buffer_map) {
    if (matcher.input1_.count(kv.second)) {
      new_buffer_map1.Set(kv.first, kv.second);
    }
  }
  new_buffer_map1.Set(new_params1.back(), matcher.intermediate_buffer_);

  PrimFunc func1 =
      PrimFunc(new_params1, body1, func->ret_type, new_buffer_map1, NullOpt, func->attrs);
  func1 = WithAttr(func1, "cutlass_kernel", String(matcher.cutlass_annotation_));
  // deal with second function
  Array<Var> new_params2;
  std::vector<int> arg_partition2;
  new_params2.push_back(Var("input", DataType::Handle()));
  for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
    Var param = func->params[i];
    if (matcher.input2_.count(func->buffer_map[param])) {
      new_params2.push_back(param);
      if (i != static_cast<int>(func->params.size()) - 1) {
        arg_partition2.push_back(i);
      }
    }
  }
  arg_partition->push_back(arg_partition2);
  Map<Var, Buffer> new_buffer_map2;
  new_buffer_map2.Set(new_params2[0], matcher.intermediate_buffer_);
  for (const auto& kv : func->buffer_map) {
    if (matcher.input2_.count(kv.second)) {
      new_buffer_map2.Set(kv.first, kv.second);
    }
  }
  PrimFunc func2 =
      PrimFunc(new_params2, body2, func->ret_type, new_buffer_map2, NullOpt, func->attrs);
  return {func1, func2};
}
}  // namespace tir

namespace relax {

// Emit 2 calls to the cutlass kernel and the rest of the function.
class SplitMutator : public ExprMutator {
 public:
  SplitMutator(const tvm::IRModule& mod) : ExprMutator(mod), mod_(mod) {}
  static IRModule Transform(const IRModule& mod) {
    SplitMutator mutator(mod);
    for (auto& kv : mod->functions) {
      if (auto* func = kv.second.as<FunctionNode>()) {
        Function new_func = Downcast<Function>(mutator(GetRef<Function>(func)));
        mutator.builder_->UpdateFunction(kv.first, new_func);
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  inline Array<Expr> GetCallTIRArgs(Expr args) {
    if (args.as<TupleNode>()) {
      return args.as<TupleNode>()->fields;
    } else {
      return {args};
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(op));
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    if (call->op.same_as(call_tir_op_)) {
      GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
      tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
      std::vector<std::vector<int>> arg_partition;
      std::pair<tir::PrimFunc, Optional<tir::PrimFunc>> split_funcs =
          tir::SplitFunctions(func, &arg_partition);
      if (!split_funcs.second.defined()) {
        ObjectPtr<CallNode> new_call = make_object<CallNode>(*call.operator->());
        GlobalVar gv = builder_->AddFunction(split_funcs.first, "cutlass_primfunc");
        new_call->args.Set(0, gv);
        return Call(new_call);
      }
      tir::PrimFunc func1 = split_funcs.first;
      tir::PrimFunc func2 = split_funcs.second.value();
      ICHECK(arg_partition.size() == 2);
      Array<Expr> args1;
      for (int p : arg_partition[0]) {
        args1.push_back(GetCallTIRArgs(call->args[1])[p]);
      }

      ShapeExpr shape1(func1->buffer_map[func1->params.back()]->shape);
      GlobalVar gv1 = builder_->AddFunction(func1, "cutlass_primfunc");
      Call call1(call_tir_op_, {gv1, Tuple(args1), shape1}, call->attrs, call->type_args);
      Var call_var1 = builder_->Emit(call1);
      Array<Expr> args2;
      args2.push_back(call_var1);
      for (int p : arg_partition[1]) {
        args2.push_back(GetCallTIRArgs(call->args[1])[p]);
      }
      GlobalVar gv2 = builder_->AddFunction(func2, "unfused_epilogue");
      Call call2(call_tir_op_, {gv2, Tuple(args2), call->args[2]}, call->attrs, call->type_args);
      return call2;
    }
    return call;
  }

  tvm::IRModule mod_;
};

namespace transform {
Pass SplitCutlass() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return SplitMutator::Transform(m); };
  return CreateModulePass(/*pass_function=*/pass_func,   //
                          /*opt_level=*/0,               //
                          /*pass_name=*/"SplitCutlass",  //
                          /*required=*/{});
}
TVM_REGISTER_GLOBAL("relax.transform.SplitCutlass").set_body_typed(SplitCutlass);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
