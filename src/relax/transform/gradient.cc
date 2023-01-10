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
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include <unordered_set>

#include "../op/make_op.h"

namespace tvm {
namespace relax {


class GradientMutator : public ExprMutator {
 public:
  explicit GradientMutator(const IRModule& mod, const GlobalVar& gvar,
                           const Array<Var>& require_grads)
      : ExprMutator(mod), mod_(mod), gvar_(std::move(gvar)), require_grads_(std::move(require_grads)) {
        CheckRequireGrads(require_grads_);
      }

  IRModule Transform() {
    IRModule new_module = GetRef<IRModule>(mod_.CopyOnWrite());

    Function new_func = CopyWithNewParams(Downcast<Function>(mod_->Lookup(gvar_)));
    Function new_func_transformed = Downcast<Function>(this->VisitExpr(new_func));

    new_module->Add(GlobalVar(gvar_->name_hint + "_adjoint"), new_func_transformed);
    return new_module;
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    ICHECK(func->body->IsInstance<SeqExprNode>());

    Expr new_body = this->VisitExpr(func->body);

    return Function(func->params, new_body, Type(), RuntimeDepShape(), func->attrs);
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) override {
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

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) override {
    builder_->BeginDataflowBlock();
    // copy bindings in the original block
    // to achieve this, we use the original VisitBinding function in ExprMutator
    // since we have overridden it, a static_cast is necessary
    for (const auto& binding : block->bindings) {
      static_cast<ExprMutator>(*this).VisitBinding(binding);
    }

    // reverse-mode ad
    for (auto it = block->bindings.rbegin(); it != block->bindings.rend(); ++it) {
      this->VisitBinding(*it);
    }

    this->Epilogue();
    return builder_->EndBlock();
  }

  void VisitBinding(const Binding& binding) override {
    // TODO(chaofan, yixin): support other types of bindings
    CHECK(binding->IsInstance<VarBindingNode>()) << "now only support VarBindingNode";
    auto var_binding = binding.as<VarBindingNode>();

    // for target_var_, generate ones op as its adjoint
    if (var_binding->var == target_var_) {
      InitGradAsOnes(var_binding->var);
    }

    if (adjoint_expr_map_.count(var_binding->var) == 0) {
      // this var is not used in the bindings handled earlier
      return;
    }

    // meet the definition of binding->var
    // create the adjoint var and bind the adjoint value to it
    Var adjoint_var = CreateAdjointVar(var_binding->var, /*is_datalfow_var=*/true);
    BindAndEmit(adjoint_var, adjoint_expr_map_[var_binding->var]);

    Expr value = var_binding->value;
    // TODO(chaofan, yixin): support other types of binding values
    CHECK(value->IsInstance<CallNode>() || value->IsInstance<TupleNode>() ||
          value->IsInstance<TupleGetItemNode>() || value->IsInstance<VarNode>() ||
          value->IsInstance<ConstantNode>())
        << "now does not support the type of binding value: " << value;

    ExprMutator::VisitBinding_(var_binding);
  }

  // handle the adjoint expr of the inputs of binding
  // for call node, we would call the registered gradient functions
  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) override {
    static const OpAttrMap<FPrimalGradient>& gradient_op_map =
        Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");

    Var adjoint_var = adjoint_var_map_[binding->var];
    const Op& call_op = GetRef<Op>(call->op.as<OpNode>());
    const Array<Expr>& partials = gradient_op_map[call_op](GetRef<Call>(call), adjoint_var);
    ICHECK(partials.size() == call->args.size()) << "partials number != inputs number";

    for (size_t i = 0; i < partials.size(); ++i) {
      UpdateAdjointForLeaf(call->args[i], partials[i]);
    }
  }

  // for Tuple nodes, we would iterate over the input tuple and update adjoint exprs for each input
  // e.g.
  // a = (b, c)
  // b_adjoint_expr += a_adjoint_var[0], c_adjoint_expr += a_adjoint_var[1]
  // a = ((b, c), d)
  // b_adjoint_expr += a_adjoint_var[0][0], c_adjoint_expr += a_adjoint_var[0][1],
  // d_adjoint_expr += a_adjoint_var[1]
  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple) override {
    UpdateAdjointForLeaf(GetRef<Tuple>(tuple), adjoint_expr_map_[binding->var]);
  }

  // for TupleGetItem nodes, we do a partial update
  // e.g.
  // b = a[0]
  // a_adjoint_expr[0] (in fields) += b_adjoint_var
  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item) override {
    ICHECK(tuple_get_item->tuple->IsInstance<VarNode>())
        << "The tuple field of a TupleGetItem is not bound to a Var";
    Type tuple_type = tuple_get_item->tuple->checked_type();
    ICHECK(tuple_type.as<TupleTypeNode>())
        << "Shape of the tuple field of a TupleGetItem must be a Tuple";
    Expr tuple_shape = tuple_get_item->tuple->shape();
    ICHECK(tuple_shape.as<TupleNode>())
        << "Type of the tuple field of a TupleGetItem must be TupleType";

    const Var& tuple_var = Downcast<Var>(tuple_get_item->tuple);
    if (adjoint_expr_map_.count(tuple_var) == 0) {
      const Tuple& init =
          BuildZerosTuple(Downcast<Tuple>(tuple_shape), Downcast<TupleType>(tuple_type));
      init->checked_type_ = tuple_var->checked_type();
      adjoint_expr_map_.Set(tuple_var, init);
    }

    ICHECK(adjoint_expr_map_[tuple_var].as<TupleNode>())
        << "Adjoint of " << tuple_var << " is expected to be a tuple";
    adjoint_expr_map_.Set(
        tuple_var, AddElementInTuple(Downcast<Tuple>(adjoint_expr_map_[tuple_var]), tuple_get_item->index,
                              adjoint_expr_map_[binding->var]));
  }

  // for assign nodes, we add the adjoint of output to the adjoint of input
  void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* var) override {
    UpdateAdjointForLeaf(GetRef<Var>(var), adjoint_expr_map_[binding->var]);
  }

  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) override {
    UpdateAdjointForLeaf(GetRef<Var>(var), adjoint_expr_map_[binding->var]);
  }

  // for constant nodes, we do not have to handle it because it does not produce adjoint
  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* var) override { return; }

 private:
  bool IsCallZeros(Expr expr) {
    static const Op& zeros = Op::Get("relax.zeros");
    if (const auto* node = expr.as<CallNode>()) {
      return node->op == zeros;
    }
    return false;
  }

  Expr ReplaceExprByVar(Expr expr) {
    if (adjoint_expr_to_var_.count(expr)) {
      return adjoint_expr_to_var_[expr];
    }
    return expr;
  }

  Var CreateAdjointVar(Var v, bool is_dataflow_var) {
    Var adjoint;
    if (is_dataflow_var) {
      adjoint = DataflowVar(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
    } else {
      adjoint = Var(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
    }
    adjoint_var_map_.Set(v, adjoint);
    return adjoint;
  }

  // Update the adjoint of leaf by partial: adjoint_expr_map_[leaf] += partial
  void UpdateAdjointForLeaf(const Expr& leaf, const Expr& partial) {
    if (const auto* node = leaf.as<VarNode>()) {
      const Var& v = GetRef<Var>(node);
      if (adjoint_expr_map_.count(v) == 0) {
        adjoint_expr_map_.Set(v, ReplaceExprByVar(partial));
      } else {
        const Expr& updated = TupleAwareAdd(adjoint_expr_map_[v], partial);
        adjoint_expr_map_.Set(v, updated);
      }
    } else if (const auto* node0 = leaf.as<TupleNode>()) {
      const auto* node1 = partial.as<TupleNode>();
      ICHECK(node1) << "Base and increment should be both tuple";
      for (size_t i = 0; i < node0->fields.size(); ++i) {
        UpdateAdjointForLeaf(node0->fields[i], node1->fields[i]);
      }
    } else if (leaf.as<ConstantNode>()) {
      // nothing to do
    } else {
      LOG(FATAL)
          << "The base is expected to a leaf node. Currently supported: Var, Tuple, Constant.";
    }
  }

  // Build a "zeros" tuple with specified shape and type
  Tuple BuildZerosTuple(const Tuple& shape, const TupleType& type) {
    Array<Expr> ret;
    for (size_t i = 0; i < shape->fields.size(); ++i) {
      if (const auto* node = shape->fields[i].as<TupleNode>()) {
        ret.push_back(BuildZerosTuple(GetRef<Tuple>(node), Downcast<TupleType>(type->fields[i])));
      } else if (shape->fields[i].as<ShapeExprNode>()) {
        const Expr& init =
            MakeZeros(shape->fields[i], Downcast<DynTensorType>(type->fields[i])->dtype);
        ret.push_back(init);
      } else {
        LOG(FATAL) << "Unsupported empty expr: " << shape->fields[i];
      }
    }
    return Tuple(ret);
  }

  // Return base + increment. A tuple-aware addition.
  Expr TupleAwareAdd(const Expr& base, const Expr& increment) {
    if (IsCallZeros(base)) {
      return ReplaceExprByVar(increment);
    } else if (IsCallZeros(increment)) {
      return ReplaceExprByVar(base);
    }

    if (const auto* base_node = base.as<TupleNode>()) {
      const TupleNode* increment_node = increment.as<TupleNode>();
      ICHECK(increment_node) << "Type not match: base and increment should be both tuple";
      ICHECK(base_node->fields.size() == increment_node->fields.size())
          << "Size of tuple not match";
      Array<Expr> result;
      for (size_t i = 0; i < base_node->fields.size(); ++i) {
        result.push_back(TupleAwareAdd(base_node->fields[i], increment_node->fields[i]));
      }
      return Tuple(result);
    } else {
      // Here we don't ReplaceExprByVar(base) because base is an adjoint_expr that
      // already bound to a adjoint_var. We wnat to update the adjoint_expr.
      return MakeAdd(base, ReplaceExprByVar(increment));
    }
  }

  // Perform an addition in a specified position of tuple.
  // e.g. tuple=(a, b, c), index=1, increment=d, then return (a, b+d, c)
  Tuple AddElementInTuple(const Tuple& tuple, int index, const Expr& increment) {
    Array<Expr> ret;
    for (size_t i = 0; i < tuple->fields.size(); ++i) {
      if ((int)i == index) {
        ret.push_back(TupleAwareAdd(tuple->fields[i], increment));
      } else {
        ret.push_back(tuple->fields[i]);
      }
    }
    return Tuple(ret);
  }

  void BindAndEmit(Var v, Expr e) {
    if (adjoint_expr_to_var_.count(e)) {
      e = adjoint_expr_to_var_[e];
    } else {
      if (!e.as<TupleNode>()) {
        adjoint_expr_to_var_.Set(e, v);
      }
    }
    if (v->IsInstance<DataflowVarNode>()) {
      builder_->Emit(VarBinding(v, e));
    } else {
      builder_->EmitOutput(VarBinding(v, e));
    }
  }

  // Init the gradient of the target_var_ and update it in adjoint_expr_map_.
  void InitGradAsOnes(const Var& var) {
    Expr ones = MakeOnes(var->shape(), Downcast<DynTensorType>(var->checked_type())->dtype);
    adjoint_expr_map_.Set(var, ones);
  }

  // Handle the return value of the AD function.
  // The return value would be like:
  // Tuple(original_return_value,
  //       Tuple(adjoint_of_require_grads_1, adjoint_of_require_grads_2, ...))
  void Epilogue() {
    // create adjoint variables for inputs, and then bind adjoints
    Array<Expr> out_adjoints;
    for (Var x : require_grads_) {
      Var new_var = this->var_remap_[x->vid];
      Var adjoint_var = CreateAdjointVar(new_var, /*is_datalfow_var=*/false);

      if (adjoint_expr_map_.count(new_var)) {
        BindAndEmit(adjoint_var, adjoint_expr_map_[new_var]);
      } else {
        BindAndEmit(
            adjoint_var,
            MakeZeros(new_var->shape(), Downcast<DynTensorType>(new_var->checked_type())->dtype));
      }
      out_adjoints.push_back(adjoint_var);
    }

    this->return_expr_ = Tuple(Array<Expr>{target_var_, Tuple(out_adjoints)});
  }

  // check that the target should be a VarNode, not DataflowVarNode
  // and a scalar of type "DynTensorType"
  static void CheckTarget(const Expr &e) {
    CHECK(e->IsInstance<VarNode>()) << "The differentiation target must be a Var";
    CHECK(!e->IsInstance<DataflowVarNode>()) << "The differentiation target is not an output node";
    CHECK(e->checked_type().as<DynTensorTypeNode>()) << "The type of the differentiation target must be DynTensorType";
    CHECK(e->shape().as<ShapeExprNode>()) << "Error when getting the shape of the differentiation target";
    CHECK(e->shape().as<ShapeExprNode>()->values.size() == 0) << "The differentiation target must be scalar";
  }

  static void CheckRequireGrads(const Array<Var> require_grads) {
    for (auto var : require_grads) {
      auto *type = var->checked_type().as<DynTensorTypeNode>();
      CHECK(type) << "The type of the input Var " << var->name_hint() << " is not DynTensorType";
      CHECK(type->dtype.is_float() || type->dtype.is_float16() || type->dtype.is_bfloat16()) << "Only Tensors of floating point dtype can require gradients, but the dtype of Var " << var->name_hint() << " is " << type->dtype;
    }
  }

  // inputs
  IRModule mod_;
  GlobalVar gvar_;
  Array<Var> require_grads_;

  // the differentiation target
  Var target_var_;
  // the return value of the differentiated function
  Expr return_expr_;

  // var to its adjoints var
  Map<Var, Var> adjoint_var_map_;
  // var to its adjoint expr
  Map<Var, Expr> adjoint_expr_map_;
  // trace binding
  Map<Expr, Var> adjoint_expr_to_var_;
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
  CHECK(func) << "relax function " << gvar->name_hint << " is not found";

  if (require_grads.defined()) {
    // there should be no duplicate var, and every var should be a parameter of the input function
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> var_set;
    for (auto var : require_grads.value()) {
      CHECK(std::find(func->params.begin(), func->params.end(), var) != func->params.end())
          << "function " << gvar->name_hint << " has no var named " << var->name_hint();
      CHECK(var_set.count(var) == 0)
          << "variable " << var->name_hint() << " appears more than once";
      var_set.emplace(var);
    }
  } else {
    // when require_grads is not specified, it would be set to all params of the function
    require_grads = func->params;
  }

  return GradientMutator(mod, gvar, require_grads.value()).Transform();
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
