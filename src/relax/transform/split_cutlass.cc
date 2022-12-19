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
#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../tir/schedule/ir_comparator.h"

namespace tvm {

const static constexpr char* kCutlassKernel = "cutlass_kernel";
const static constexpr char* kCutlassCodegen = "cutlass_codegen";
const static constexpr char* kCSource = "c_source";
const static constexpr char* kCSourceFmt = "c_source_fmt";
const static constexpr char* kCSourceFmtCuda = "cu";

namespace tir {

class BlockMatcher : public TensorizeComparator {
 public:
  using SymbolMap = std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
  explicit BlockMatcher(const tir::PrimFunc& pattern, const Array<Var>& pattern_vars)
      : TensorizeComparator(IRModule({{GlobalVar(""), pattern}}), false), pattern_(pattern) {
    for (const auto& pattern_var : pattern_vars) {
      this->pattern_vars_.insert(pattern_var);
    }
    this->evaluated_symbols.push_back(SymbolMap());
  }

  bool Match(const For& top) {
    const ForNode* pattern_top = pattern_->body.as<BlockRealizeNode>()->block->body.as<ForNode>();
    ICHECK(pattern_top) << "Invalid pattern function";
    if (!VisitStmt(top, GetRef<Stmt>(pattern_top))) return false;
    // Get evaluated symbols, buffers from the pattern.
    for (const auto& arg : pattern_->params) {
      auto it = pattern_->buffer_map.find(arg);
      if (it != pattern_->buffer_map.end()) {
        auto itt = rhs_buffer_map_.find((*it).second);
        ICHECK(itt != rhs_buffer_map_.end());
        evaluated_buffers.push_back(itt->second);
      }
    }
    return true;
  }

  std::vector<SymbolMap> evaluated_symbols;
  std::vector<Buffer> evaluated_buffers;

 private:
  Optional<PrimExpr> QueryEvaluatedSymbols(const Var& var) {
    for (const SymbolMap& symbol_map : evaluated_symbols) {
      auto it = symbol_map.find(var);
      if (it != symbol_map.end()) {
        return it->second;
      }
    }
    return NullOpt;
  }

  bool VisitExpr(const PrimExpr& lhs, const PrimExpr& rhs) final {
    if (const auto* op = rhs.as<VarNode>()) {
      if (pattern_vars_.count(GetRef<Var>(op))) {
        // special case for pattern vars
        const auto* lhs_ptr = lhs.as<VarNode>();
        if (lhs_ptr == nullptr) {
          if (lhs->IsInstance<tir::IntImmNode>() || lhs->IsInstance<tir::FloatImmNode>()) {
            Optional<PrimExpr> value = QueryEvaluatedSymbols(GetRef<Var>(op));
            if (value.defined()) {
              if (!analyzer_.CanProveEqual(lhs, value.value())) return false;
            } else {
              evaluated_symbols.back()[GetRef<Var>(op)] = lhs;
            }
            return true;
          } else {
            return false;
          }
        }
      }
    }
    // pattern_var * expr
    if (const auto* rhs_ptr = rhs.as<MulNode>()) {
      const auto* operand_a = rhs_ptr->a.as<VarNode>();
      const auto* operand_b = rhs_ptr->b.as<VarNode>();
      if (operand_a != nullptr && pattern_vars_.count(GetRef<Var>(operand_a))) {
        // pattern var is on the left
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->b);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_a)] = MakeConstScalar(rhs_ptr->b.dtype(), 1);
          return true;
        }
      }
      if (operand_b != nullptr && pattern_vars_.count(GetRef<Var>(operand_b))) {
        // pattern var is on the right
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->a);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_b)] = MakeConstScalar(rhs_ptr->a.dtype(), 1);
          return true;
        }
      }
    }
    // pattern_Var + expr
    if (const auto* rhs_ptr = rhs.as<AddNode>()) {
      const auto* operand_a = rhs_ptr->a.as<VarNode>();
      const auto* operand_b = rhs_ptr->b.as<VarNode>();
      if (operand_a != nullptr && pattern_vars_.count(GetRef<Var>(operand_a))) {
        // pattern var is on the left
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->b);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_a)] = MakeConstScalar(rhs_ptr->b.dtype(), 0);
          return true;
        }
      }
      if (operand_b != nullptr && pattern_vars_.count(GetRef<Var>(operand_b))) {
        // pattern var is on the right
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->a);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_b)] = MakeConstScalar(rhs_ptr->a.dtype(), 0);
          return true;
        }
      }
    }
    return TensorizeComparator::VisitExpr(lhs, rhs);
  }

  bool VisitExpr_(const tir::AddNode* add, const PrimExpr& other) final {
    const auto* rhs = other.as<AddNode>();
    if (rhs == nullptr) return false;
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(add->a, rhs->a) && VisitExpr(add->b, rhs->b);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(add->a, rhs->b) && VisitExpr(add->b, rhs->a);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    return false;
  }

  bool VisitExpr_(const tir::MulNode* mul, const PrimExpr& other) final {
    const auto* rhs = other.as<MulNode>();
    if (rhs == nullptr) return false;
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(mul->a, rhs->a) && VisitExpr(mul->b, rhs->b);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(mul->a, rhs->b) && VisitExpr(mul->b, rhs->a);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    return false;
  }

  bool VisitExpr_(const tir::CallNode* call, const PrimExpr& other) final {
    const auto* rhs = other.as<CallNode>();
    if (rhs == nullptr) return false;
    const auto* lhs_op = call->op.as<OpNode>();
    const auto* rhs_op = rhs->op.as<OpNode>();
    if (lhs_op == nullptr || rhs_op == nullptr) return false;
    if (lhs_op->name != rhs_op->name) return false;
    if (call->args.size() != rhs->args.size()) return false;
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (!VisitExpr(call->args[i], rhs->args[i])) return false;
    }
    return true;
  }

  bool VisitStmt_(const tir::ForNode* op, const Stmt& other) final {
    const auto* rhs = other.as<ForNode>();
    loop_stack_lhs_.push_back(GetRef<For>(op));
    loop_stack_rhs_.push_back(GetRef<For>(rhs));
    // The body of loop must be loop or BlockRealize
    if (!op->body->IsInstance<BlockRealizeNode>() && !op->body->IsInstance<ForNode>()) {
      return false;
    }
    if (!rhs->body->IsInstance<BlockRealizeNode>() && !rhs->body->IsInstance<ForNode>()) {
      return false;
    }
    // Build mapping between the loop vars
    if (!DefEqual(op->loop_var, rhs->loop_var)) return false;
    // Only handle the case where the loop start from 0
    if (!is_zero(op->min) || !is_zero(rhs->min)) return false;
    if (op->thread_binding.defined() || rhs->thread_binding.defined()) return false;
    if (op->kind != ForKind::kSerial || op->kind != rhs->kind) return false;
    if (!op->annotations.empty() || !rhs->annotations.empty()) return false;
    // Match the extents of loops
    if (!VisitExpr(op->extent, rhs->extent)) return false;
    return VisitStmt(op->body, rhs->body);
  }

  bool VisitStmt_(const tir::BlockNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockNode>();
    // Check block equality.
    // All iter vars and buffer regions including the order should match.
    // When checking iter vars, DefEqual is used to remap variables.
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &BlockMatcher::CompareIterVar)) {
      return false;
    }
    // disallow alloc buffers inside the block
    if (!op->alloc_buffers.empty() || !rhs->alloc_buffers.empty()) return false;
    if (!CompareArray(op->writes, rhs->writes, &BlockMatcher::CompareBufferRegion)) {
      return false;
    }
    if (!CompareArray(op->reads, rhs->reads, &BlockMatcher::CompareBufferRegion)) {
      return false;
    }
    // The body of the block has to be BufferStore
    if (!op->body->IsInstance<BufferStoreNode>() || !rhs->body->IsInstance<BufferStoreNode>()) {
      return false;
    }
    // Handle init block
    if (op->init.defined() && !rhs->init.defined()) return false;
    if (!op->init.defined() && rhs->init.defined()) return false;
    if (op->init.defined() && rhs->init.defined()) {
      if (!VisitStmt(op->init.value(), rhs->init.value())) return false;
    }
    return VisitStmt(op->body, rhs->body);
  }

  bool VisitStmt_(const BlockRealizeNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockRealizeNode>();
    // Only allow trivial bindings
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      if (!op->iter_values[i].same_as(loop_stack_lhs_[i]->loop_var)) return false;
    }
    for (size_t i = 0; i < rhs->iter_values.size(); ++i) {
      if (!rhs->iter_values[i].same_as(loop_stack_rhs_[i]->loop_var)) return false;
    }
    // Disallow predicates now
    if (!is_one(op->predicate) || !is_one(rhs->predicate)) return false;
    return VisitStmt(op->block, rhs->block);
  }

  bool VisitStmt_(const BufferStoreNode* op, const Stmt& other) {
    const auto* rhs = other.as<BufferStoreNode>();
    return CompareBufferAccess(op, rhs) && VisitExpr(op->value, rhs->value);
  }

  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<BufferLoadNode>();
    return CompareBufferAccess(op, rhs);
  }

  bool CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
    if (lhs.same_as(rhs)) return true;
    auto it = rhs_buffer_map_.find(rhs);
    bool equal;
    if (it != rhs_buffer_map_.end()) {
      equal = (*it).second.same_as(lhs);
    } else {
      // Compare shape
      if (lhs->shape.size() != rhs->shape.size()) return false;
      for (size_t i = 0; i < lhs->shape.size(); ++i) {
        if (!VisitExpr(lhs->shape[i], rhs->shape[i])) return false;
      }
      // Remap both buffer itself and buffer data
      equal =
          DefEqual(lhs->data, rhs->data) && lhs->dtype == rhs->dtype && lhs.scope() == rhs.scope();
      if (equal) {
        rhs_buffer_map_[rhs] = lhs;
      }
    }
    return equal;
  }

  bool CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs) {
    if (!CompareBuffer(lhs->buffer, rhs->buffer)) {
      return false;
    }
    return CompareArray(lhs->region, rhs->region, &BlockMatcher::CompareRange);
  }

  template <typename T>
  bool CompareBufferAccess(const T* lhs, const T* rhs) {
    if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
    return CompareArray(lhs->indices, rhs->indices, &BlockMatcher::VisitExpr);
  }

  template <typename T, typename Self, typename F>
  bool CompareArray(const Array<T>& lhs, const Array<T>& rhs, F Self::*cmp) {
    if (lhs.same_as(rhs)) return true;
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!(static_cast<Self*>(this)->*cmp)(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  arith::Analyzer analyzer_;
  std::vector<For> loop_stack_lhs_, loop_stack_rhs_;
  tir::PrimFunc pattern_;
  std::unordered_set<Var, ObjectHash, ObjectEqual> pattern_vars_;
};

// Analyze the function and match it with supported cutlass kernels
class FuncMatcher : public StmtExprVisitor {
 public:
  void Match(Stmt body) {
    OpMatternMatch(body);
    if (fail) return;
    auto f = TypedPackedFunc<Array<ObjectRef>(Array<Map<Var, PrimExpr>>, Array<Array<Buffer>>,
                                              Array<runtime::String>)>(
        *tvm::runtime::Registry::Get("tvm.relax.cutlass.op_pattern_stitch"));
    ICHECK(f != nullptr) << "Cannot find cutlass op pattern stitch function";
    Array<ObjectRef> op_stitich_result =
        f(Array<Map<Var, PrimExpr>>(evaluated_symbols_), Array<Array<Buffer>>(evaluated_buffers_),
          Array<runtime::String>(matched_pattern_names_));
    ICHECK_EQ(op_stitich_result.size(), 2);
    num_matched_ops = op_stitich_result[1].as<IntImmNode>()->value;
    if (num_matched_ops == 0) {
      fail = true;
      return;
    }
    cutlass_kernel_code = Downcast<runtime::String>(op_stitich_result[0]);
    this->VisitStmt(body);
  }

  // alloc_buffers for the first function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs1;
  // alloc_buffers for the second function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs2;
  // whether the current block is in the first function
  Map<Block, Bool> block_partition;
  // input buffers for the first function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input1;
  // input buffers for the second function
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input2;
  // The output buffer for the first function, which is also the input buffer for the second
  // function
  Buffer intermediate_buffer;
  // The number of matched ops in the function
  size_t num_matched_ops = 0;
  // The CUDA code for the cutlass kernel
  runtime::String cutlass_kernel_code;
  // Indicate whether we have failed. If failed, we will not do any further analysis and directly
  // return the original one.
  bool fail = false;

 private:
  // Find an op that matches this block
  bool BlockPatternMatch(const For& top) {
    auto f = tvm::runtime::Registry::Get("tvm.relax.cutlass.get_op_pattern_list");
    auto g = tvm::runtime::Registry::Get("tvm.relax.cutlass.get_op_pattern_func");
    auto h = tvm::runtime::Registry::Get("tvm.relax.cutlass.get_op_pattern_vars");
    CHECK(f != nullptr) << "Cannot find tvm.relax.cutlass.get_op_pattern_list";
    CHECK(g != nullptr) << "Cannot find tvm.relax.cutlass.get_op_pattern_func";
    CHECK(h != nullptr) << "Cannot find tvm.relax.cutlass.get_op_pattern_vars";
    Array<runtime::String> pattern_list = (*f)();

    for (const runtime::String& pattern : pattern_list) {
      tir::PrimFunc pattern_func = (*g)(pattern);
      Array<Var> pattern_vars = (*h)(pattern);
      BlockMatcher block_matcher(pattern_func, pattern_vars);
      if (block_matcher.Match(top)) {
        // We have found a match
        evaluated_symbols_.push_back(block_matcher.evaluated_symbols.back());
        evaluated_buffers_.push_back(block_matcher.evaluated_buffers);
        matched_pattern_names_.push_back(pattern);
        return true;
      }
    }
    // The block fails to match any pattern
    return false;
  }

  // For each block in the body, try to find its corresponding pattern one by one
  void OpMatternMatch(const Stmt& body) {
    Array<Stmt> blocks;
    if (body->IsInstance<ForNode>()) {
      // {for}
      blocks = {body};
    } else if (const SeqStmtNode* seq = body.as<SeqStmtNode>()) {
      blocks = seq->seq;
    } else {
      fail = true;
      return;
    }
    for (const Stmt& stmt : blocks) {
      const ForNode* loop = stmt.as<ForNode>();
      if (loop == nullptr || !BlockPatternMatch(GetRef<For>(loop))) {
        break;
      }
    }
    if (matched_pattern_names_.empty()) fail = true;
  }

  void VisitStmt_(const BlockNode* op) final {
    block_counter_++;
    bool is_matching_ = block_counter_ <= num_matched_ops;
    if (block_counter_ == num_matched_ops) {
      allocs1.erase(intermediate_buffer);
    }
    for (const auto& read : op->reads) {
      if (is_matching_) {
        input1.insert(read->buffer);
      } else {
        input2.insert(read->buffer);
      }
    }
    for (const auto& write : op->writes) {
      if (is_matching_) {
        allocs1.insert(write->buffer);
      } else if (allocs1.count(write->buffer)) {
        fail = true;
        return;
      } else {
        allocs2.insert(write->buffer);
      }
      if (is_matching_) {
        intermediate_buffer = write->buffer;
      } else {
        input2.insert(write->buffer);
      }
    }
    block_partition.Set(GetRef<Block>(op), Bool(is_matching_));
  }

  size_t block_counter_ = 0;

  std::vector<Map<Var, PrimExpr>> evaluated_symbols_;
  std::vector<Array<Buffer>> evaluated_buffers_;
  std::vector<runtime::String> matched_pattern_names_;
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
      : block_partition(block_partition), allocs_(allocs), is_cutlass_(is_cutlass) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*block.operator->());
    if (op->name_hint != "root") {
      ICHECK(block_partition.count(GetRef<Block>(op)));
      bool block_is_cutlass = block_partition[GetRef<Block>(op)]->value;
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
  Map<Block, Bool> block_partition;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs_;
  bool is_cutlass_ = false;
};

/*!
 * \brief Split the input function into two functions, one for the cutlass kernel and one for the
 * rest.
 * \param func The input function.
 * \param arg_partition The input arg for the functions after split.
 * \return A pair of functions, the first one is the cutlass kernel and the second one is the
 * rest.
 */
std::pair<PrimFunc, Optional<PrimFunc>> SplitFunctions(
    PrimFunc func, std::vector<std::vector<int>>* arg_partition) {
  // Step 1. Find the cutlass kernel and the rest.
  FuncMatcher matcher;
  matcher.Match(func->body.as<BlockRealizeNode>()->block->body);
  if (matcher.fail) {
    return {func, NullOpt};
  }
  bool has_second_func = false;
  for (const auto& pr : matcher.block_partition) {
    if (!pr.second->value) {
      has_second_func = true;
      break;
    }
  }
  if (!has_second_func) {
    // No need to split the function.
    return {WithAttr(func, kCutlassKernel, matcher.cutlass_kernel_code), NullOpt};
  }
  // Step 2. Split the function into two functions.
  Stmt body1 = BlockMasker::Mask(func->body, matcher.block_partition, matcher.allocs1, true);
  Stmt body2 = BlockMasker::Mask(func->body, matcher.block_partition, matcher.allocs2, false);
  // Step 3. Craft the first function.
  Array<Var> new_params1;
  std::vector<int> arg_partition1;
  for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
    Var param = func->params[i];
    if (matcher.input1.count(func->buffer_map[param])) {
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
    if (matcher.input1.count(kv.second)) {
      new_buffer_map1.Set(kv.first, kv.second);
    }
  }
  new_buffer_map1.Set(new_params1.back(), matcher.intermediate_buffer);
  PrimFunc func1 = PrimFunc(new_params1, body1, func->ret_type, new_buffer_map1, func->attrs);
  func1 = WithAttr(func1, kCutlassKernel, matcher.cutlass_kernel_code);
  // Step 4. Craft the second function.
  Array<Var> new_params2;
  std::vector<int> arg_partition2;
  new_params2.push_back(Var("input", DataType::Handle()));
  for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
    Var param = func->params[i];
    if (matcher.input2.count(func->buffer_map[param])) {
      new_params2.push_back(param);
      if (i != static_cast<int>(func->params.size()) - 1) {
        arg_partition2.push_back(i);
      }
    }
  }
  arg_partition->push_back(arg_partition2);
  Map<Var, Buffer> new_buffer_map2;
  new_buffer_map2.Set(new_params2[0], matcher.intermediate_buffer);
  for (const auto& kv : func->buffer_map) {
    if (matcher.input2.count(kv.second)) {
      new_buffer_map2.Set(kv.first, kv.second);
    }
  }
  PrimFunc func2 = PrimFunc(new_params2, body2, func->ret_type, new_buffer_map2, func->attrs);
  return {func1, func2};
}
}  // namespace tir

namespace relax {
void StringReplace(std::string* subject, const std::string& search, const std::string& replace) {
  for (size_t pos = 0; (pos = subject->find(search, pos)) != std::string::npos;
       pos += replace.length()) {
    subject->replace(pos, search.length(), replace);
  }
}

tvm::BaseFunc CodegenWithCutlass(const tir::PrimFuncNode* pf, String global_symbol) {
  using namespace tvm::tir;
  Optional<runtime::String> cutlass_kernel_code =
      pf->attrs.GetAttr<runtime::String>(kCutlassKernel);
  if (!cutlass_kernel_code.defined()) {
    return GetRef<tir::PrimFunc>(pf);
  }
  std::string source = cutlass_kernel_code.value();
  StringReplace(&source, "{global_symbol}", global_symbol);
  ExternFunc ret(global_symbol);
  ret = WithAttrs(std::move(ret), Map<String, ObjectRef>{
                                      {String(kCSource), String(source)},
                                      {String(kCSourceFmt), String(kCSourceFmtCuda)},
                                  });
  return ret;
}

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
      // the first argument is the function to be called
      GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
      // retrieve the function from the module and split it
      tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
      std::vector<std::vector<int>> arg_partition;
      // split the function into two functions, one for the cutlass kernel and one for the rest.
      std::pair<tir::PrimFunc, Optional<tir::PrimFunc>> split_funcs =
          tir::SplitFunctions(func, &arg_partition);
      if (!split_funcs.second.defined()) {
        // no need to split, the function itself a cutlass kernel
        // emit the call to the cutlass kernel
        ObjectPtr<CallNode> new_call = make_object<CallNode>(*call.operator->());
        // builder_->UpdateFunction(gv, split_funcs.first);
        builder_->UpdateFunction(gv, CodegenWithCutlass(split_funcs.first.get(), gv->name_hint));
        return Call(new_call);
      }
      tir::PrimFunc func1 = tir::RenewDefs(split_funcs.first);
      tir::PrimFunc func2 = tir::RenewDefs(split_funcs.second.value());
      ICHECK(arg_partition.size() == 2);
      // emit the first call to the cutlass kernel
      Array<Expr> args1;
      for (int p : arg_partition[0]) {
        args1.push_back(GetCallTIRArgs(call->args[1])[p]);
      }
      ShapeExpr shape1(func1->buffer_map[func1->params.back()]->shape);
      // replace the function in the module with the cutlass kernel
      GlobalVar gv1 = builder_->AddFunction(func1, "cutlass_primfunc");
      builder_->UpdateFunction(gv1, CodegenWithCutlass(func1.get(), gv1->name_hint));
      tir::Buffer intermediate_buffer = func1->buffer_map.at(func1->params.back());
      DataType dtype = intermediate_buffer->dtype;
      Call call1(call_tir_op_, {gv1, Tuple(args1), shape1}, call->attrs,
                 {DynTensorType(intermediate_buffer->shape.size(), dtype)});
      Var call_var1 = builder_->Emit(call1);
      // emit the second call to the rest of the function
      Array<Expr> args2;
      args2.push_back(call_var1);
      for (int p : arg_partition[1]) {
        args2.push_back(GetCallTIRArgs(call->args[1])[p]);
      }
      GlobalVar gv2 = builder_->AddFunction(func2, "unfused_epilogue");
      Call call2(call_tir_op_, {gv2, Tuple(args2), call->args[2]}, call->attrs, call->type_args);
      builder_->UpdateFunction(gv, WithoutAttr(func, "global_symbol"));
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
