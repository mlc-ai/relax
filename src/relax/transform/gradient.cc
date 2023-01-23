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
 * \file src/relax/transform/gradient.cc
 * \brief Reverse-mode automatic differentiation.
 *
 * Now only supports differentiating one function in the IRModule with one dataflow block
 * with respect to the only return value of the function, which needs to be scalar.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include <unordered_set>

#include "../op/tensor/binary.h"
#include "../op/tensor/create.h"
#include "utils.h"

namespace tvm {
namespace relax {

using AdjointMsg = NestedMsg<Expr>;

// A tool class for GradientMutator
// Visit the forward bindings and generate the backward bindings
class BackwardBindingGenerator : public ExprVisitor {
 public:
  /*!
   * \brief Generate the backward bindings for the corresponding GradientMutator
   *
   * \param builder The BlockBuilder of GradientMutator, used to generate bindings
   * \param forward_block The forward DataflowBlock
   * \param require_grads The Var list to differentiate w.r.t.
   * \param target_var The target Var to differentiate
   * \return The return expr of new adjoint function.
   */
  static Expr Generate(const BlockBuilder& builder, const DataflowBlock& forward_block,
                       const Array<Var>& require_grads, const Var& target_var) {
    BackwardBindingGenerator generator;
    generator.builder_ = builder;

    // Initialize the adjoint of target_var as ones op. We have already check the target.
    auto target_sinfo = GetStructInfoAs<TensorStructInfoNode>(target_var);
    const Expr& target_adjoint = ones(target_sinfo->shape.value(), target_sinfo->dtype);
    UpdateStructInfo(target_adjoint, GetRef<StructInfo>(target_sinfo));
    generator.adjoint_msg_map_.Set(target_var, AdjointMsg(target_adjoint));

    // We do reverse-mode ad, so visit bindings backwards
    for (auto it = forward_block->bindings.rbegin(); it != forward_block->bindings.rend(); ++it) {
      generator.VisitBinding(*it);
    }

    return generator.Epilogue(require_grads, target_var);
  }

 private:
  void VisitBinding(const Binding& binding) final {
    // TODO(chaofan, yixin): support other types of bindings
    CHECK(binding->IsInstance<VarBindingNode>()) << "now only support VarBindingNode";
    auto var_binding = binding.as<VarBindingNode>();

    auto it = adjoint_msg_map_.find(var_binding->var);
    if (it == adjoint_msg_map_.end()) {
      // This var is not used in the following bindings
      return;
    }

    // Meet the definition of binding->var
    // Create the adjoint var and bind the adjoint value to it
    EmitAdjoint(var_binding->var, (*it).second, true);

    Expr value = var_binding->value;
    // TODO(chaofan, yixin): support other types of binding values
    CHECK(value->IsInstance<CallNode>() || value->IsInstance<TupleNode>() ||
          value->IsInstance<TupleGetItemNode>() || value->IsInstance<VarNode>() ||
          value->IsInstance<ConstantNode>())
        << "now does not support the type of binding value: " << value;

    ExprVisitor::VisitBinding_(var_binding);
  }

  // Handle the adjoint expr of the inputs of binding
  // For call node, we would call the registered gradient functions
  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) final {
    static const OpAttrMap<FPrimalGradient>& gradient_op_map =
        Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");

    Var adjoint_var = adjoint_var_map_[binding->var];
    const Op& call_op = GetRef<Op>(call->op.as<OpNode>());
    const Array<Expr>& partials =
        gradient_op_map[call_op](binding->var, GetRef<Call>(call), adjoint_var, builder_);
    ICHECK(partials.size() == call->args.size()) << "partials number != inputs number";

    for (size_t i = 0; i < partials.size(); ++i) {
      Expr partial = partials[i];
      if (!partial->struct_info_.defined()) {
        UpdateStructInfo(partial, GetStructInfo(call->args[i]));
      }
      UpdateAdjoint(call->args[i], partial);
    }
  }

  // For Tuple nodes, we would iterate over the input tuple and update adjoint exprs for each input
  // e.g.
  // a = (b, c)
  // b_adjoint += a_adjoint_var[0], c_adjoint += a_adjoint_var[1]
  // a = ((b, c), d)
  // b_adjoint += a_adjoint_var[0][0], c_adjoint += a_adjoint_var[0][1],
  // d_adjoint += a_adjoint_var[1]
  //
  // Here we use adjoint_var to simplify calculation
  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple) final {
    UpdateAdjoint(GetRef<Tuple>(tuple), adjoint_var_map_[binding->var]);
  }

  // For TupleGetItem nodes, we do a partial update
  // e.g.
  // b = a[0]
  // a_adjoint[0] += b_adjoint_var
  // If a_adjoint does not exist, we would create a zeros tuple as a_adjoint first, and then add
  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* tuple_get_item) final {
    ICHECK(tuple_get_item->tuple->IsInstance<VarNode>())
        << "The tuple field of a TupleGetItem is not bound to a Var";
    auto tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(tuple_get_item->tuple);
    ICHECK(tuple_sinfo != nullptr)
        << "The tuple field of a TupleGetItem must has a TupleStructInfo";

    const Var& tuple_var = Downcast<Var>(tuple_get_item->tuple);
    if (adjoint_msg_map_.count(tuple_var) == 0) {
      const AdjointMsg& init = InitZerosAdjointNested(GetRef<StructInfo>(tuple_sinfo));
      adjoint_msg_map_.Set(tuple_var, init);
    }

    adjoint_msg_map_.Set(tuple_var,
                         AddInAdjointMsg(adjoint_msg_map_[tuple_var], tuple_get_item->index,
                                         ExprToAdjointMsg(adjoint_var_map_[binding->var])));
  }

  // For assign nodes, we add the adjoint of output to the adjoint of input
  void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* var) final {
    UpdateAdjoint(GetRef<Var>(var), adjoint_var_map_[binding->var]);
  }

  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) final {
    UpdateAdjoint(GetRef<Var>(var), adjoint_var_map_[binding->var]);
  }

  // For constant nodes, we do not have to handle it because it does not contribute to the adjoint
  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* var) final { return; }

  bool IsCallZeros(const Expr& expr) {
    return expr->IsInstance<CallNode>() && Downcast<Call>(expr)->op == Op::Get("relax.zeros");
  }

  // Add partial (Expr type) to the adjoint of expr
  void UpdateAdjoint(const Expr& expr, const Expr& partial) {
    DecomposeNestedMsg(expr, ExprToAdjointMsg(partial), [&](Expr leaf, AdjointMsg msg) {
      if (leaf->IsInstance<VarNode>()) {
        const Var& v = Downcast<Var>(leaf);
        if (adjoint_msg_map_.count(v) == 0) {
          adjoint_msg_map_.Set(v, msg);
        } else {
          adjoint_msg_map_.Set(v, TupleAwareAdd(adjoint_msg_map_[v], msg));
        }
      } else if (leaf->IsInstance<ConstantNode>()) {
        // nothing to do
      } else {
        LOG(FATAL) << "UpdateAdjoint: leaf type not supported. Currently Var and Constant leaves "
                      "are supported.";
      }
    });
  }

  // Create a zeros AdjointMsg with specified struct info
  // When sinfo is TupleStructInfo, we would create a nested zeros Tuple
  static AdjointMsg InitZerosAdjointNested(const StructInfo& sinfo) {
    return MapToNestedMsg<Expr>(sinfo, [](StructInfo sinfo) {
      auto tensor_sinfo = sinfo.as<TensorStructInfoNode>();
      ICHECK(tensor_sinfo) << "The leaf of adjoint should be a Tensor.";
      ICHECK(tensor_sinfo->shape.defined()) << "Error: missing shape when building zeros tuple.";
      const Expr& init = zeros(tensor_sinfo->shape.value(), tensor_sinfo->dtype);
      UpdateStructInfo(init, sinfo);
      return init;
    });
  }

  // Return base + increment. A tuple-aware addition.
  AdjointMsg TupleAwareAdd(const AdjointMsg& base, const AdjointMsg& increment) {
    return CombineNestedMsg(base, increment, [&](Expr lhs, Expr rhs) {
      if (IsCallZeros(lhs)) {
        return rhs;
      } else if (IsCallZeros(rhs)) {
        return lhs;
      }
      auto sinfo = GetStructInfoAs<TensorStructInfoNode>(lhs);
      ICHECK(sinfo) << "The leaf of adjoint should have StructInfo and be a Tensor.";
      ICHECK(GetStructInfoAs<TensorStructInfoNode>(rhs))
          << "The leaf of adjoint should have StructInfo and be a Tensor.";
      Expr res = add(lhs, rhs);
      UpdateStructInfo(res, GetRef<StructInfo>(sinfo));
      return res;
    });
  }

  // Perform an addition in a specified position of tuple.
  // e.g. tuple=(a, b, c), index=1, increment=d, then return (a, b+d, c)
  AdjointMsg AddInAdjointMsg(const AdjointMsg& adjoint, int index, const AdjointMsg& increment) {
    ICHECK(adjoint.IsNested()) << "The adjoint should be nested.";
    Array<AdjointMsg> arr = adjoint.NestedArray();
    arr.Set(index, TupleAwareAdd(arr[index], increment));
    return AdjointMsg(arr);
  }

  Expr AdjointMsgToExpr(AdjointMsg msg) {
    return MapFromNestedMsg<Expr>(msg, [](AdjointMsg leaf_msg) { return leaf_msg.LeafValue(); });
  }

  AdjointMsg ExprToAdjointMsg(Expr expr) {
    return MapToNestedMsgBySInfo<Expr>(expr, [](Expr leaf) {
      ICHECK(GetStructInfoAs<TensorStructInfoNode>(leaf))
          << "The leaf of adjoint: " << leaf << " should have StructInfo and be a Tensor.";
      return AdjointMsg(leaf);
    });
  }

  // Transform the adjoint expressed as NestedMsg<Expr> into adjoint Expr, and then emit it
  // If the adjoint is assigned to a DataflowVar (the adjoint corresponds to a non-output binding),
  // it would be stored in adjoint_var_map_ for future lookup
  Var EmitAdjoint(const Var& source_var, const AdjointMsg& adjoint, bool is_dataflow_var) {
    Var adjoint_var;
    if (is_dataflow_var) {
      adjoint_var = builder_->Emit(AdjointMsgToExpr(adjoint), source_var->name_hint() + "_adjoint");
      adjoint_var_map_.Set(source_var, adjoint_var);
    } else {
      adjoint_var =
          builder_->EmitOutput(AdjointMsgToExpr(adjoint), source_var->name_hint() + "_adjoint");
    }
    return adjoint_var;
  }

  // Handle the return value of the AD function.
  // Returns the new return value, which would be like:
  // Tuple(original_return_value,
  //       Tuple(adjoint_of_require_grads_1, adjoint_of_require_grads_2, ...))
  Expr Epilogue(const Array<Var>& require_grads, const Var& target_var) {
    // create adjoint variables for inputs, and then bind adjoints
    Array<Expr> out_adjoints;

    for (Var var : require_grads) {
      // If the var don't have adjoint msg, it do not contribute to the target
      // so its adjoint is zeros
      AdjointMsg adjoint =
          adjoint_msg_map_.Get(var).value_or(InitZerosAdjointNested(GetStructInfo(var)));
      Var adjoint_var = EmitAdjoint(var, adjoint, false);
      out_adjoints.push_back(adjoint_var);
    }

    return Tuple({target_var, Tuple(out_adjoints)});
  }

  // The block builder of the corresponding GradientMutator, to emit bindings
  BlockBuilder builder_;

  // Forward Var to its adjoint Var
  Map<Var, Var> adjoint_var_map_;
  // Forward Var to its adjoint NestedMsg<Expr>
  // We use NestedMsg<Expr> to save the adjoint information (equivalent to adjoint Expr)
  // When emitting, adjoint information will be transformed into adjoint Expr
  Map<Var, AdjointMsg> adjoint_msg_map_;
};

class GradientMutator : public ExprMutator {
 public:
  static IRModule Transform(IRModule mod, GlobalVar gvar, Array<Var> require_grads) {
    Function old_func = Downcast<Function>(mod->Lookup(gvar));
    CheckRequireGrads(require_grads, old_func->params, gvar->name_hint);

    Function new_func = CopyWithNewParams(old_func);
    // map the parameter list into new params
    for (size_t i = 0; i < require_grads.size(); ++i) {
      int idx = std::find(old_func->params.begin(), old_func->params.end(), require_grads[i]) -
                old_func->params.begin();
      require_grads.Set(i, new_func->params[idx]);
    }

    GradientMutator mutator(mod, require_grads);
    Function new_func_transformed = Downcast<Function>(mutator.VisitExpr(new_func));

    IRModule new_module = GetRef<IRModule>(mod.CopyOnWrite());
    new_module->Add(GlobalVar(gvar->name_hint + "_adjoint"), new_func_transformed);
    return new_module;
  }

  Expr VisitExpr_(const FunctionNode* func) final {
    CHECK(func->body->IsInstance<SeqExprNode>())
        << "Currently the body of the function must be SeqExpr.";
    auto* func_sinfo = GetStructInfoAs<FuncStructInfoNode>(GetRef<Function>(func));
    CHECK(func_sinfo->params.defined()) << "Currently don't support opaque function.";

    Expr new_body = this->VisitExpr(func->body);

    return Function(func->params, new_body, GetStructInfo(return_expr_), func->attrs);
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) final {
    // TODO(chaofan, yixin): multiple blocks AD
    CHECK(seq_expr->blocks.size() == 1) << "now only support one dataflow block";
    // TODO(chaofan, yixin): AD in non-dataflow block.
    CHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockNode>())
        << "now only support one dataflow block";

    // the return value should be a VarNode, and a scalar
    CheckTarget(seq_expr->body);
    this->target_var_ = Downcast<Var>(seq_expr->body);

    BindingBlock new_block = this->VisitBindingBlock(seq_expr->blocks[0]);
    return SeqExpr({new_block}, this->return_expr_);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    builder_->BeginDataflowBlock();
    // accept bindings in the original block
    for (const auto& binding : block->bindings) {
      this->VisitBinding(binding);
    }

    // generate backward bindings and the return value
    return_expr_ = BackwardBindingGenerator::Generate(this->builder_, GetRef<DataflowBlock>(block),
                                                      this->require_grads_, this->target_var_);

    return builder_->EndBlock();
  }

 private:
  GradientMutator(const IRModule& module, const Array<Var>& require_grads)
      : ExprMutator(module), require_grads_(require_grads) {}

  // check that the target should be a output Var
  // and a scalar Tensor
  static void CheckTarget(const Expr& e) {
    CHECK(e->IsInstance<VarNode>() && !e->IsInstance<DataflowVarNode>())
        << "The differentiation target must be a output Var";
    CHECK(IsScalarTensor(e)) << "The differentiation target must be a scalar Tensor, but the "
                                "StructInfo of the given target "
                             << e << " is " << GetStructInfo(e);
  }

  // Check every Var in require_grads:
  // 1. there should be no duplicate var
  // 2. every var should be a parameter of the function
  // 3. the type of the input var should be Tensor of floating point dtype, or Tuple of that
  static void CheckRequireGrads(const Array<Var>& require_grads, const Array<Var>& func_params,
                                const String& func_name) {
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> var_set;
    for (const auto& var : require_grads) {
      CHECK(std::find(func_params.begin(), func_params.end(), var) != func_params.end())
          << "There is no Var named " << var->name_hint() << " in the parameters of the function "
          << func_name;
      CHECK_EQ(var_set.count(var), 0) << "Var " << var->name_hint() << " appears more than once";
      var_set.emplace(var);

      auto is_float_tensor = [](const TensorStructInfo& sinfo) {
        return sinfo->dtype.is_float() || sinfo->dtype.is_float16() || sinfo->dtype.is_bfloat16();
      };

      CHECK(IsNestedTensorConditioned(GetStructInfo(var), is_float_tensor))
          << "Only Tensors of floating point dtype or Tuples of float "
             "Tensors can require gradients, but the StructInfo of Var "
          << var->name_hint() << " is " << GetStructInfo(var);
    }
  }

  // differentiation sources
  Array<Var> require_grads_;
  // the differentiation target
  Var target_var_;

  // the return value of the differentiated function
  Expr return_expr_;
};

/*!
 * \brief This is the internal function of tvm::relax::transform::Gradient.
 * \param mod The module
 * \param gvar The GlobalVar of the specified function
 * \param require_grads The relax variables whose adjoints are needed.
 * \return The module after transformation.
 */
IRModule Gradient(const IRModule& mod, const GlobalVar& gvar, Optional<Array<Var>> require_grads) {
  auto* func = mod->Lookup(gvar).as<FunctionNode>();
  CHECK(func) << "Relax function " << gvar->name_hint << " is not found";

  if (!require_grads.defined()) {
    // when require_grads is not specified, it would be set to all params of the function
    require_grads = func->params;
  }

  return GradientMutator::Transform(mod, gvar, require_grads.value());
}

namespace transform {

Pass Gradient(GlobalVar global_var, Optional<Array<Var>> require_grads) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return relax::Gradient(mod, global_var, require_grads); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"Gradient",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.Gradient").set_body_typed(Gradient);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
