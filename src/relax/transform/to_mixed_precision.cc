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
 * \file src/relax/transform/to_mixed_precision.cc
 * \brief Automatic mixed precision pass.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include "infer_amp_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

using runtime::String;

int GetMixedPrecisionInfo(const CallNode* call_node) {
  const OpNode* op_node = call_node->op.as<OpNode>();
  if (op_node == nullptr) {
    return -1;
  }
  Op op = GetRef<Op>(op_node);
  auto attr_map = Op::GetAttrMap<TMixedPrecisionPolicy>("TMixedPrecisionPolicy");
  return attr_map.count(op) ? attr_map[op] : MixedPrecisionPolicyKind::kNever;
}

/*!
 * \brief Main logic to automatically cast fp32 input modules to fp16 for certain ops.
 *
 * Structurally speaking, a Relax function is composed of a series of VarBinding and
 * MatchCast. And a specific class of VarBindings is the basic unit we want to rewrite.
 * Formally, they are of the form:
 *
 * var = Call(Op, [args], attrs)
 *
 * where Op is a specific op we want to rewrite, and attrs is the attributes of the op.
 * var and args are all exprs with type Tensor or Tuple of Tensors. They might
 * be vars, constants, or Tuple of vars and constants.
 * Depending on the properties of the op, we may have 3 different ways to rewrite it:
 *
 * 1. kAlways: Always cast the args to fp16
 *    Currently, this is only used for gemm and conv ops (to favor the use of TensorCore)
 *    We always cast the input args to fp16, and the dtype of the accumulator is configured
 *    by the global output_dtype parameter (default to fp32). Finally we will cast the
 *    accumulator back to fp16 (if the output_dtype is fp32).
 *
 * 2. kFollow: If the output is expected to be fp16 by any of the consumers
 *    of the op, we will cast the args to fp16. Otherwise, we will cast the args to fp32.
 *
 * 3. kNever: Never cast the args to fp16. Always cast all args to fp32 (the original dtype).
 *    Some ops, such as softmax, have numerical issues when using fp16. We will always use fp32
 *    to ensure the correctness.
 *
 * DTypeDecisionCollector:
 *   We will first use a backward propagation pass (since kFollow requires the
 *   knowledge of the consumer to notify its producers) to know the expected dtype of each var. Note
 *   that multiple dtypes of a var might be expected by different consumers. We keep the lowest
 *   precision required as the dtype to realize the unique copy of the var.
 *
 * ToMixedPrecisionRewriter:
 *   We will then use a forward propagation pass to rewrite the program. Since we only keep one
 *   specific data type for each var, and we will cast the var to the required dtype when we
 *   encounter it locally if needed. Note that we may cast the var to some certain dtype multiple
 *   times, but we decide not to store and reuse the casted copy due to the storage concern and to
 *   be more friendly to inlining and operator fusion.
 *
 * The information of each op is registered in the
 * Op::GetAttr<FInferMixedPrecision>("FInferMixedPrecision"). The registered function has signature:
 * FInferMixedPrecision. We will call the registered function with the original call and the global
 * output_dtype parameter. The registered function will return the policy of the op, whether the op
 * can adjust the dtype of the accumulator, and the new call node with output_dtype set to the
 * global output_dtype parameter.
 *
 * Key design: wrap_param op
 *   We need to use fp16 parameters (which appear as constants in the program), but the type
 *   inference will fail if some parameters are fp16 and some are fp32 in the original module. To
 *   solve this, we introduce a new op wrap_param, which will wrap the original parameter and cast
 *   it to fp32 var.
 *
 *   When we encounter the var afterwards, we will directly replace it with the parameter. This
 *   information is tracked by the const_map_.
 */
class DTypeDecisionCollector : public ExprVisitor {
 public:
  explicit DTypeDecisionCollector(DataType output_dtype) : output_dtype_(output_dtype) {}

  static VarDTypeMap Collect(Function func, DataType output_dtype) {
    DTypeDecisionCollector collector(output_dtype);
    collector.VisitExpr(func);
    return std::move(collector.var_dtype_map_);
  }

 private:
  // merge the message for a var
  void UpdateVarDTypeMap(const Var& var, const NType& dtype) {
    auto it = var_dtype_map_.find(var);
    if (it == var_dtype_map_.end()) {
      var_dtype_map_[var] = dtype;
    } else {
      var_dtype_map_[var] = NTypeMerge(it->second, dtype);
    }
  }

  // merge the message for all vars in the expr list
  void RequireArgsToType(Array<Expr> args, Array<NType> to) {
    ICHECK(args.size() == to.size()) << "Invalid target dtypes";
    for (size_t i = 0; i < args.size(); ++i) {
      auto fvisitleaf = [&](const Expr& expr, NType to) {
        if (const auto* var = expr.as<VarNode>()) {
          UpdateVarDTypeMap(GetRef<Var>(var), to);
        } else if (expr->IsInstance<ConstantNode>()) {
          // Constant can be casted anyway, so we don't need to do anything here
          return;
        } else {
          LOG(FATAL) << "Unsupported argument type: " << expr->GetTypeKey();
        }
      };
      DecomposeNestedMsg(args[i], to[i], fvisitleaf);
    }
  }

  // merge the message for all vars in the expr list
  void RequireArgsToType(Array<Expr> args, DataType to) {
    std::vector<NType> to_arr;
    for (const Expr& arg : args) {
      // Note that this implicitly requires all arg to be nested tensors.
      to_arr.push_back(NTypeFrom(arg, to));
    }
    RequireArgsToType(args, to_arr);
  }

  void VisitVars_(const VarNode* op) {
    Var var = GetRef<Var>(op);
    if (IsNestedTensor(var)) {
      // require the var to be its original dtype
      UpdateVarDTypeMap(var, NTypeFrom(var));
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) final { VisitVars_(op); }

  void VisitExpr_(const DataflowVarNode* op) final { VisitVars_(op); }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    auto policy = GetMixedPrecisionInfo(call_node);
    if (policy == -1) {
      ExprVisitor::VisitBinding_(binding, call_node);
      return;
    }
    if (policy == kAlways) {
      // require inputs to be fp16
      RequireArgsToType(call_node->args, fp16_);
    } else if (policy == kFollow) {
      // require inputs to be fp16 if the output is required to be fp16, otherwise fp32
      auto it = var_dtype_map_.find(binding->var);
      if (it == var_dtype_map_.end()) return;
      NType all_fp32 = NTypeFrom(binding->var, fp32_);
      RequireArgsToType(call_node->args, NTypeEqual()(it->second, all_fp32) ? fp32_ : fp16_);
    } else if (policy == kNever) {
      // require inputs to be fp32 (the original dtype)
      RequireArgsToType(call_node->args, fp32_);
    } else {
      LOG(FATAL) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }
  }

  // override the following methods to visit in backward order
  void VisitExpr_(const SeqExprNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->body);
    for (auto it = op->blocks.rbegin(); it != op->blocks.rend(); it++) {
      this->VisitBindingBlock(*it);
    }

    if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
      this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
    }
  }

  void VisitBindingBlock_(const BindingBlockNode* block) {
    for (auto it = block->bindings.rbegin(); it != block->bindings.rend(); it++) {
      this->VisitBinding(*it);
    }
  }

  void VisitBindingBlock_(const DataflowBlockNode* block) {
    for (auto it = block->bindings.rbegin(); it != block->bindings.rend(); it++) {
      this->VisitBinding(*it);
    }
  }

  void VisitExpr_(const IfNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->true_branch);
    this->VisitExpr(op->false_branch);
    this->VisitExpr(op->cond);

    if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
      this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
    }
  }

  DataType fp16_ = DataType(DataType::TypeCode::kFloat, 16, 1);
  DataType fp32_ = DataType(DataType::TypeCode::kFloat, 32, 1);
  DataType output_dtype_;
  VarDTypeMap var_dtype_map_;
};

class ToMixedPrecisionRewriter : public ExprMutator {
 public:
  explicit ToMixedPrecisionRewriter(const VarDTypeMap* var_dtype_map, DataType output_dtype)
      : var_dtype_map_(var_dtype_map), output_dtype_(output_dtype) {}

 private:
  using VarMap = std::unordered_map<Id, Var, ObjectPtrHash, ObjectPtrEqual>;

  Var GetRealized(const Var& var) {
    auto it = var_remap_.find(var->vid);
    return it == var_remap_.end() ? var : it->second;
  }

  // TODO(@bohan): implements some postorder function accepts a visitor closure
  class VarReplacer : public ExprMutator {
   public:
    explicit VarReplacer(const ToMixedPrecisionRewriter::VarMap& var_map) : var_map_(var_map) {}

    static Expr Replace(const Expr& expr, const ToMixedPrecisionRewriter::VarMap& var_map) {
      VarReplacer replacer(var_map);
      return replacer(expr);
    }

   private:
    Expr VisitExpr_(const VarNode* op) final {
      auto it = var_map_.find(op->vid);
      return it == var_map_.end() ? GetRef<Var>(op) : it->second;
    }

    const ToMixedPrecisionRewriter::VarMap& var_map_;
  };

  // Util function to rewrite the expr to the given dtype
  // we first replace the vars with the realized vars
  // then rewrite each leaf tensor to the given dtype if necessary
  // Note that this function only accepts expr with nested tensor type
  Expr RewriteExpr(const Expr& expr, const NType& to) {
    auto fvisitleaf = [&](const Expr& expr, std::array<NType, 1> to) -> Expr {
      const auto* tensor = GetStructInfoAs<TensorStructInfoNode>(expr);
      ICHECK(tensor != nullptr) << "Only support rewriting tensor expr";
      if (NTypeEqual()(to[0], NTypeFrom(expr))) return expr;
      return astype(expr, DataType(String2DLDataType(to[0].LeafValue())));
    };
    return TransformTupleLeaf<String>(VarReplacer::Replace(expr, var_remap_),
                                      std::array<NType, 1>({to}), fvisitleaf);
  }

  Array<Expr> RewriteArgs(const Array<Expr>& args, DataType to) {
    Array<Expr> new_args;
    for (const Expr& arg : args) {
      if (IsNestedTensor(arg)) {
        new_args.push_back(RewriteExpr(arg, NTypeFrom(arg, to)));
      } else {
        new_args.push_back(arg);
      }
    }
    return new_args;
  }

  void RealizeVarDef(const Var& var) {
    // realize the var to the required dtype if necessary
    auto it = var_dtype_map_->find(var);
    if (it == var_dtype_map_->end()) return;
    // Rewrite current realized var to the required dtype
    Var cur_realize = GetRealized(var);
    Expr rewrite = RewriteExpr(cur_realize, it->second);
    // If cur_realize is not rewritten, we don't need to emit a new var
    if (!rewrite.same_as(cur_realize)) {
      // Emit a new var, and update the var remap
      var_remap_[var->vid] = builder_->Emit(rewrite);
    }
  }

  void VisitBinding(const Binding& binding) final {
    // Emit the binding
    ExprMutator::VisitBinding(binding);
    // For every binding, we try to realize the var to the required dtype
    RealizeVarDef(binding->var);
  }

  Expr VisitVar_(const Var& var) {
    auto it = var_dtype_map_->find(var);
    if (it == var_dtype_map_->end()) return ExprMutator::VisitExpr_(var.get());
    // We rewrite the realized var to the original dtype
    return RewriteExpr(var, NTypeFrom(var));
  }

  Expr VisitExpr_(const VarNode* op) final { return VisitVar_(GetRef<Var>(op)); }

  Expr VisitExpr_(const DataflowVarNode* op) final { return VisitVar_(GetRef<Var>(op)); }

  Expr VisitExpr_(const FunctionNode* op) final {
    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : op->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      all_params_unchanged &= param.same_as(new_param);
    }

    // Override here to realize the params, and build a binding block
    builder_->BeginBindingBlock();
    for (const auto& param : op->params) {
      RealizeVarDef(param);
    }
    BindingBlock bb = builder_->EndBlock();

    Expr body = this->VisitWithNewScope(op->body, params);
    if (!bb->bindings.empty()) {
      if (const auto* seq = body.as<SeqExprNode>()) {
        Array<BindingBlock> new_blocks = seq->blocks;
        new_blocks.insert(new_blocks.begin(), bb);
        body = SeqExpr(new_blocks, seq->body);
      } else {
        body = SeqExpr({bb}, body);
      }
      body = builder_->Normalize(body);
    }

    // FuncStructInfo does not depend on Expr
    if (all_params_unchanged && body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      return Function(params, body, op->ret_struct_info, op->attrs);
    }
    return ExprMutator::VisitExpr_(op);
  }

  // Detect if all the args are realized to fp16
  bool AllFp16Available(const Array<Expr>& args) {
    // TODO(@bohan): implements a tuple visitor that accepts a leaf handler to replace this logic
    for (const Expr& arg : args) {
      // arg is either a var or a constant or a tuple
      if (const auto* var_node = arg.as<VarNode>()) {
        // arg is a var
        Var realized = GetRealized(GetRef<Var>(var_node));
        NType realized_sinfo = NTypeFrom(realized);
        NType realized_fp16 = NTypeFrom(realized, fp16_);
        if (!NTypeEqual()(realized_fp16, realized_sinfo)) {
          return false;
        }
      } else if (const auto* tuple_node = arg.as<TupleNode>()) {
        if (!AllFp16Available(tuple_node->fields)) {
          return false;
        }
      }
      // constant can always be casted, so we don't need to check it
    }
    return true;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) override {
    auto it = var_dtype_map_->find(binding->var);
    if (it == var_dtype_map_->end()) {
      // we don't have dtype decision for this var
      ExprMutator::VisitBinding_(binding, call_node);
      return;
    }
    auto policy = GetMixedPrecisionInfo(call_node);
    if (policy == -1) {
      // not an op call
      ExprMutator::VisitBinding_(binding, call_node);
      return;
    }
    const auto* op_node = call_node->op.as<OpNode>();
    ICHECK(op_node != nullptr);
    Op op = GetRef<Op>(op_node);
    if (wrap_param_op.same_as(op)) {
      // wrap_param
      const auto* constant = call_node->args[0].as<ConstantNode>();
      ICHECK(constant != nullptr) << "Invalid wrap_param: " << GetRef<Call>(call_node);
      ReEmitBinding(binding, GetRef<Expr>(constant));
      return;
    }
    DataType to;
    ObjectPtr<CallNode> new_call = make_object<CallNode>(*call_node);
    if (policy == kAlways) {
      to = fp16_;
      auto attr_map = Op::GetAttrMap<FInferMixedPrecision>("FInferMixedPrecision");
      ICHECK(attr_map.count(op));
      auto f = attr_map[op];
      new_call = make_object<CallNode>(*(f(GetRef<Call>(call_node), output_dtype_).get()));
    } else if (policy == kFollow) {
      NType all_fp32 = NTypeFrom(binding->var, fp32_);
      to = NTypeEqual()(it->second, all_fp32) ? fp32_ : fp16_;
    } else if (policy == kNever) {
      to = fp32_;
    } else {
      LOG(FATAL) << "Unsupported TMixedPrecisionPolicy: " << policy;
    }
    new_call->args = std::move(RewriteArgs(call_node->args, to));
    new_call->struct_info_ = NullOpt;
    ReEmitBinding(binding, builder_->Normalize(Call(new_call)));
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) override {
    // var = (v0, v1, ...)
    auto it = var_dtype_map_->find(binding->var);
    if (it != var_dtype_map_->end()) {
      ReEmitBinding(binding, builder_->Normalize(RewriteExpr(GetRef<Tuple>(val), it->second)));
      return;
    }
    // we don't have dtype decision for this tuple
    ExprMutator::VisitBinding_(binding, val);
  }

  const VarDTypeMap* var_dtype_map_;

  DataType fp16_ = DataType(DataType::TypeCode::kFloat, 16, 1);
  DataType fp32_ = DataType(DataType::TypeCode::kFloat, 32, 1);
  DataType output_dtype_;

  const Op& wrap_param_op = Op::Get("relax.wrap_param");
};

Expr ToMixedPrecision(const Function& f, const DataType& out_dtype) {
  VarDTypeMap var_dtype_map = std::move(DTypeDecisionCollector::Collect(f, out_dtype));
  ToMixedPrecisionRewriter mutator(&var_dtype_map, out_dtype);
  return mutator(f);
}

namespace transform {

Pass ToMixedPrecisionPass(const DataType& out_dtype) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ToMixedPrecision(f, out_dtype));
      };
  return CreateFunctionPass(pass_func, 0, "ToMixedPrecision", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ToMixedPrecision").set_body_typed(ToMixedPrecisionPass);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
