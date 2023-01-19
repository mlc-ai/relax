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

namespace tvm {
namespace relax {

using AdjointType = NestedMsg<Expr>;

// A tool class for GradientMutator
// Visit the forward bindings and generate the backward bindings
class BackwardBindingGenerator : public ExprVisitor {
 public:
  /*!
   * \brief Generate the backward bindings for the corresponding GradientMutator
   *
   * \param forward_block The forward dataflow block
   * \param require_grads The relax variables whose adjoints are needed.
   * \param post The expression with rewritten inputs.
   */
  Expr Generate(const BlockBuilder& builder, const DataflowBlock& forward_block,
                const Array<Var>& require_grads, const Var& target_var) {
    this->builder_ = builder;
    this->target_var_ = target_var;

    // We do reverse-mode ad, so visit bindings backwards
    for (auto it = forward_block->bindings.rbegin(); it != forward_block->bindings.rend(); ++it) {
      this->VisitBinding(*it);
    }

    return this->Epilogue(require_grads);
  }

  void VisitBinding(const Binding& binding) override {
    // TODO(chaofan, yixin): support other types of bindings
    CHECK(binding->IsInstance<VarBindingNode>()) << "now only support VarBindingNode";
    auto var_binding = binding.as<VarBindingNode>();

    // For target_var_, generate ones op as its adjoint
    if (var_binding->var == target_var_) {
      InitGradAsOnes(var_binding->var);
    }

    if (adjoint_expr_map_.count(var_binding->var) == 0) {
      // This var is not used in the bindings handled earlier
      return;
    }

    // Meet the definition of binding->var
    // Create the adjoint var and bind the adjoint value to it
    Var adjoint_var = CreateAdjointVar(var_binding->var, /*is_datalfow_var=*/true);
    BindAndEmit(adjoint_var, adjoint_expr_map_[var_binding->var]);

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
  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) override {
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
      UpdateAdjointForLeaf(call->args[i], ExprToNestedMsg(partial));
    }
  }

  // For Tuple nodes, we would iterate over the input tuple and update adjoint exprs for each input
  // e.g.
  // a = (b, c)
  // b_adjoint_expr += a_adjoint_var[0], c_adjoint_expr += a_adjoint_var[1]
  // a = ((b, c), d)
  // b_adjoint_expr += a_adjoint_var[0][0], c_adjoint_expr += a_adjoint_var[0][1],
  // d_adjoint_expr += a_adjoint_var[1]
  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple) override {
    UpdateAdjointForLeaf(GetRef<Tuple>(tuple), ExprToNestedMsg(adjoint_var_map_[binding->var]));
  }

  // For TupleGetItem nodes, we do a partial update
  // e.g.
  // b = a[0]
  // a_adjoint_expr[0] (in fields) += b_adjoint_var
  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item) override {
    ICHECK(tuple_get_item->tuple->IsInstance<VarNode>())
        << "The tuple field of a TupleGetItem is not bound to a Var";
    auto tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(tuple_get_item->tuple);
    ICHECK(tuple_sinfo != nullptr)
        << "The tuple field of a TupleGetItem must has a TupleStructInfo";

    const Var& tuple_var = Downcast<Var>(tuple_get_item->tuple);
    if (adjoint_expr_map_.count(tuple_var) == 0) {
      const AdjointType& init = BuildZerosAdjoint(tuple_sinfo);
      adjoint_expr_map_.Set(tuple_var, init);
    }

    adjoint_expr_map_.Set(
        tuple_var, AddElementInNestedAdjoint(adjoint_expr_map_[tuple_var], tuple_get_item->index,
                                             ExprToNestedMsg(adjoint_var_map_[binding->var])));
  }

  // For assign nodes, we add the adjoint of output to the adjoint of input
  void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* var) override {
    UpdateAdjointForLeaf(GetRef<Var>(var), ExprToNestedMsg(adjoint_var_map_[binding->var]));
  }

  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) override {
    UpdateAdjointForLeaf(GetRef<Var>(var), ExprToNestedMsg(adjoint_var_map_[binding->var]));
  }

  // For constant nodes, we do not have to handle it because it does not produce adjoint
  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* var) override { return; }

 private:
  bool IsCallZeros(const Expr& expr) {
    return expr->IsInstance<CallNode>() && Downcast<Call>(expr)->op == Op::Get("relax.zeros");
  }

  Var CreateAdjointVar(const Var& v, bool is_dataflow_var) {
    Var adjoint;
    auto sinfo = GetStructInfo(v);
    if (is_dataflow_var) {
      adjoint = DataflowVar(v->name_hint() + "_adjoint", sinfo);
    } else {
      adjoint = Var(v->name_hint() + "_adjoint", sinfo);
    }
    adjoint_var_map_.Set(v, adjoint);
    return adjoint;
  }

  // Update the adjoint of a leaf node by partial: adjoint_expr_map_[leaf] += partial
  // Here leaf node means Var, Tuple, or Constant
  // The arguments of 1) *Function calls*, or 2) *Tuples* can be any kind of leaf nodes,
  // so we handle them uniformly here
  void UpdateAdjointForLeaf(const Expr& leaf, const AdjointType& partial) {
    if (const auto* node = leaf.as<VarNode>()) {
      const Var& v = GetRef<Var>(node);
      if (adjoint_expr_map_.count(v) == 0) {
        adjoint_expr_map_.Set(v, partial);
      } else {
        adjoint_expr_map_.Set(v, TupleAwareAdd(adjoint_expr_map_[v], partial));
      }
    } else if (const auto* node0 = leaf.as<TupleNode>()) {
      ICHECK(partial.IsNested()) << "If leaf is Tuple, the adjoint must be nested in thie case.";
      auto arr = partial.NestedArray();
      ICHECK(node0->fields.size() == arr.size());
      for (size_t i = 0; i < node0->fields.size(); ++i) {
        UpdateAdjointForLeaf(node0->fields[i], arr[i]);
      }
    } else if (leaf.as<ConstantNode>()) {
      // nothing to do
    } else {
      LOG(FATAL)
          << "The base is expected to a leaf node. Currently supported: Var, Tuple, Constant.";
    }
  }

  // Build a "zeros" tuple with specified tuple struct info and type
  AdjointType BuildZerosAdjoint(const TupleStructInfoNode* sinfo) {
    return MapToNestedMsg<Expr>(GetRef<StructInfo>(sinfo), [](StructInfo sinfo) {
      auto tensor_sinfo = sinfo.as<TensorStructInfoNode>();
      ICHECK(tensor_sinfo) << "The leaf of adjoint should be a Tensor.";
      ICHECK(tensor_sinfo->shape.defined()) << "Error: missing shape when building zeros tuple.";
      const Expr& init = zeros(tensor_sinfo->shape.value(), tensor_sinfo->dtype);
      UpdateStructInfo(init, sinfo);
      return init;
    });
  }

  // Return base + increment. A tuple-aware addition.
  AdjointType TupleAwareAdd(const AdjointType& base, const AdjointType& increment) {
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
  AdjointType AddElementInNestedAdjoint(const AdjointType& adjoint, int index,
                                        const AdjointType& increment) {
    ICHECK(adjoint.IsNested()) << "The adjoint should be nested.";
    Array<AdjointType> arr = adjoint.NestedArray();
    Array<AdjointType> new_arr;
    new_arr.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
      if (static_cast<int>(i) == index) {
        new_arr.push_back(TupleAwareAdd(arr[i], increment));
      } else {
        new_arr.push_back(arr[i]);
      }
    }
    return AdjointType(new_arr);
  }

  Expr NestedMsgToExpr(AdjointType msg) {
    auto fmapmerge = [](Array<Expr> subexpr) { return Tuple(subexpr); };
    auto fmapleaf = [](AdjointType leaf_msg) {
      ICHECK(leaf_msg.IsLeaf());
      return leaf_msg.LeafValue();
    };
    auto fmapnull = []() {
      LOG(FATAL) << "Error: null field should not appear in adjoint.";
      return Expr();  // just for type checking
    };
    return MapFromNestedMsg<Expr>(msg, fmapmerge, fmapleaf, fmapnull);
  }

  AdjointType ExprToNestedMsg(Expr expr) {
    auto fmapleaf = [](Expr leaf) {
      auto sinfo = GetStructInfoAs<TensorStructInfoNode>(leaf);
      ICHECK(sinfo) << "The leaf of adjoint: " << leaf
                    << " should have StructInfo and be a Tensor.";
      return AdjointType(leaf);
    };
    auto fstepprocess = [](Expr expr, size_t index) {
      if (const auto* node = expr.as<TupleNode>()) {
        return node->fields[index];
      }
      const auto* sinfo = GetStructInfoAs<TupleStructInfoNode>(expr);
      ICHECK(sinfo != nullptr) << "GetField: expr must has TupleStructInfo.";
      Expr ret = TupleGetItem(expr, index);
      UpdateStructInfo(ret, sinfo->fields[index]);
      return ret;
    };
    return MapToNestedMsg<Expr>(expr, fstepprocess, fmapleaf);
  }

  void BindAndEmit(const Var& v, const AdjointType& adjoint) {
    Expr adjoint_expr = NestedMsgToExpr(adjoint);
    builder_->EmitNormalized(VarBinding(v, builder_->Normalize(adjoint_expr)));
  }

  // Init the gradient of the target_var_ and update it in adjoint_expr_map_.
  void InitGradAsOnes(const Var& var) {
    auto sinfo = GetStructInfoAs<TensorStructInfoNode>(var);
    ICHECK(sinfo->shape.defined());
    adjoint_expr_map_.Set(
        var, AdjointType(builder_->Normalize(ones(sinfo->shape.value(), sinfo->dtype))));
  }

  // Handle the return value of the AD function.
  // Returns the new return value, which would be like:
  // Tuple(original_return_value,
  //       Tuple(adjoint_of_require_grads_1, adjoint_of_require_grads_2, ...))
  Expr Epilogue(const Array<Var>& require_grads) {
    // create adjoint variables for inputs, and then bind adjoints
    Array<Expr> out_adjoints;

    for (Var var : require_grads) {
      Var adjoint_var = CreateAdjointVar(var, /*is_datalfow_var=*/false);

      if (adjoint_expr_map_.count(var)) {
        BindAndEmit(adjoint_var, adjoint_expr_map_[var]);
      } else {
        // here we could assert that var do not contribute to the target
        // so we emit zeros as adjoint here
        if (auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(var)) {
          ICHECK(sinfo->shape.defined());
          BindAndEmit(adjoint_var, AdjointType(zeros(sinfo->shape.value(), sinfo->dtype)));
        } else if (auto* sinfo = GetStructInfoAs<TupleStructInfoNode>(var)) {
          BindAndEmit(adjoint_var, BuildZerosAdjoint(sinfo));
        }
      }

      out_adjoints.push_back(adjoint_var);
    }

    Expr ret = Tuple(Array<Expr>{target_var_, Tuple(out_adjoints)});
    return builder_->Normalize(ret);
  }

  // The block builder of the corresponding GradientMutator, to emit bindings
  BlockBuilder builder_;

  // The differentiation target
  Var target_var_;

  // Forward Var to its adjoint Var
  Map<Var, Var> adjoint_var_map_;
  // Forward Var to its adjoint Expr
  Map<Var, AdjointType> adjoint_expr_map_;
};

class GradientMutator : public ExprMutator {
 public:
  explicit GradientMutator(const IRModule& mod, const GlobalVar& gvar,
                           const Array<Var>& require_grads)
      : ExprMutator(mod),
        mod_(mod),
        gvar_(std::move(gvar)),
        require_grads_(std::move(require_grads)) {}

  IRModule Transform() {
    Function old_func = Downcast<Function>(mod_->Lookup(gvar_));
    CheckRequireGrads(require_grads_, old_func->params, gvar_->name_hint);

    Function new_func = CopyWithNewParams(old_func);
    // map the parameter list into new params
    for (size_t i = 0; i < require_grads_.size(); ++i) {
      int idx = std::find(old_func->params.begin(), old_func->params.end(), require_grads_[i]) -
                old_func->params.begin();
      require_grads_.Set(i, new_func->params[idx]);
    }

    Function new_func_transformed = Downcast<Function>(this->VisitExpr(new_func));

    IRModule new_module = GetRef<IRModule>(mod_.CopyOnWrite());
    new_module->Add(GlobalVar(gvar_->name_hint + "_adjoint"), new_func_transformed);
    return new_module;
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    CHECK(func->body->IsInstance<SeqExprNode>())
        << "Currently the body of the function must be SeqExpr.";
    auto* func_sinfo = GetStructInfoAs<FuncStructInfoNode>(GetRef<Function>(func));
    CHECK(func_sinfo->params.defined()) << "Currently don't support opaque function.";

    Expr new_body = this->VisitExpr(func->body);

    return Function(func->params, new_body, GetStructInfo(return_expr_), func->attrs);
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
    // accept bindings in the original block
    for (const auto& binding : block->bindings) {
      this->VisitBinding(binding);
    }

    // generate backward bindings and the return value
    return_expr_ = BackwardBindingGenerator().Generate(this->builder_, GetRef<DataflowBlock>(block),
                                                       this->require_grads_, this->target_var_);

    return builder_->EndBlock();
  }

 private:
  // check that the target should be a VarNode, not DataflowVarNode
  // and a scalar of type "DynTensorType"
  static void CheckTarget(const Expr& e) {
    CHECK(e->IsInstance<VarNode>()) << "The differentiation target must be a Var";
    CHECK(!e->IsInstance<DataflowVarNode>()) << "The differentiation target is not an output node";
    auto sinfo = GetStructInfoAs<TensorStructInfoNode>(e);
    CHECK(sinfo != nullptr) << "The differentiation target must be a Tensor";
    CHECK(sinfo->shape.defined() && sinfo->shape.as<ShapeExprNode>())
        << "Error when getting the shape of the differentiation target";
    CHECK(sinfo->shape.as<ShapeExprNode>()->values.size() == 0)
        << "The differentiation target must be scalar";
  }

  // Check the type of the input var should be Tensor of floating point dtype, or Tuple of that
  static void CheckFloatTensorType(const StructInfo& sinfo, const String& name_hint) {
    if (auto* tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
      for (auto item : tuple_sinfo->fields) {
        CheckFloatTensorType(item, name_hint);
      }
    } else if (auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
      CHECK(tensor_sinfo->dtype.is_float() || tensor_sinfo->dtype.is_float16() ||
            tensor_sinfo->dtype.is_bfloat16())
          << "Only Tensors of floating point dtype can require gradients, but the dtype of Var "
          << name_hint << " is " << tensor_sinfo->dtype;
    } else {
      LOG(FATAL) << "The input Var " << name_hint << " is neither a Tensor nor a Tuple of Tensors";
    }
  }

  // Checks every Var in require_grads:
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
      CHECK(var_set.count(var) == 0) << "Var " << var->name_hint() << " appears more than once";
      var_set.emplace(var);

      CheckFloatTensorType(GetStructInfo(var), var->name_hint());
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
