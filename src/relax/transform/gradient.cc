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

#include <unordered_set>

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include "../op/make_op.h"


namespace tvm {
namespace relax {

// TODO(chaofan, yixin): support constants
class GradientMutator : public ExprMutator {
 public:
  explicit GradientMutator(IRModule mod, const Array<Var>& require_grads)
      : ExprMutator(std::move(mod)), require_grads_(std::move(require_grads)) {}

  Expr VisitExpr_(const FunctionNode* func) override {
    // copy the parameters and set var_remap_
    Array<Var> new_params;
    for (Var param : func->params) {
      Var new_param = Var(param->vid, param->shape(), param->checked_type(), param->span);
      this->var_remap_[param->vid] = new_param;
      new_params.push_back(new_param);
    }

    // use VisitExpr to remap the variables in the input body
    CHECK(func->body->IsInstance<SeqExprNode>()) << "the body of the function is not SeqExprNode. Please use relax.transform.Normalize to normalize the function first.";
    Expr body_with_remapped_var = static_cast<ExprMutator>(*this).VisitExpr(func->body);

    // AD transformation
    Expr new_body = this->VisitExpr(body_with_remapped_var);

    return Function(new_params, new_body, Type(), RuntimeDepShape(), func->attrs);
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) override {
    // TODO(chaofan, yixin): multiple blocks AD
    CHECK(seq_expr->blocks.size() == 1) << "now only support one dataflow block";
    // TODO(chaofan, yixin): AD in non-dataflow block.
    CHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockNode>()) << "now only support one dataflow block";

    // the return value should be a VarNode, and a scalar
    CheckTarget(seq_expr->body);
    this->target_var_ = Downcast<Var>(seq_expr->body);

    BindingBlock new_block = this->VisitBindingBlock(seq_expr->blocks[0]);
    return SeqExpr({new_block}, this->return_expr_);
  }


  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) override {
    builder_->BeginDataflowBlock();
    // copy bindings in the original block
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

  // handle return values of the AD function
  // the return value would be like:
  // Tuple(original_return_value, Tuple(adjoint_of_require_grads_1, adjoint_of_require_grads_2, ...))
  void Epilogue() {
    // create adjoint var for inputs
    for (size_t i = 0; i < new_params.size(); ++i) {
      if (std::find(require_grads_.begin(), require_grads_.end(), func->params[i]) !=
          require_grads_.end()) {
        CreateAdjointVar(new_params[i], /*is_datalfow_var=*/false);
      } else {
        CreateAdjointVar(new_params[i], /*is_datalfow_var=*/true);
      }
    }
    // handle the return values and types
    Array<Expr> out_expr, out_adjoints;
    out_expr.push_back(seq_expr->body);

    // emit the input adjoints
    static const Op& default_op = Op::Get("relax.zeros");
    for (size_t i = 0; i < new_params.size(); ++i) {
      if (std::find(require_grads_.begin(), require_grads_.end(), func->params[i]) != require_grads_.end()) {
        const Var& adjoint_var = adjoint_var_map_[new_params[i]];
        if (adjoint_expr_map_.count(new_params[i])) {
          BindAndEmit(adjoint_var, adjoint_expr_map_[new_params[i]]);
        } else {
          ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
          auto type = Downcast<DynTensorType>(new_params[i]->checked_type());
          attrs->dtype = type->dtype;

          const Expr& default_adjoint = Call(default_op, {new_params[i]->shape()}, Attrs(attrs));
          BindAndEmit(adjoint_var, default_adjoint);
        }
        out_adjoints.push_back(adjoint_var);
      }
    }

    out_expr.push_back(Tuple(out_adjoints));
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
    CHECK(value->IsInstance<CallNode>() || value->IsInstance<TupleNode>() || value->IsInstance<TupleGetItemNode>() || value->IsInstance<VarNode>())
      << "now does not support the type of binding value: " << value;

    ExprMutator::VisitBinding_(var_binding);
  }

  // handle the adjoint expr of the inputs of binding
  // for call node, we would call the registered gradient functions
  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) override {
    static const OpAttrMap<FPrimalGradient>& gradient_op_map =
        Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");

    Var adjoint_var = adjoint_var_map_[binding->var];
    Op call_op = GetRef<Op>(call->op.as<OpNode>());
    Array<Expr> partials = gradient_op_map[call_op](GetRef<Call>(call), adjoint_var);
    ICHECK(partials.size() == call->args.size()) << "partials number != inputs number";

    for (size_t i = 0; i < partials.size(); ++i) {
      const VarNode* arg = call->args[i].as<VarNode>();
      ICHECK(arg != nullptr);
      // TODO(chaofan, yixin): support Ops taking tuple input or returning tuple output
      UpdateExprMap(GetRef<Var>(arg), partials[i]);
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
    UpdateExprMap(GetRef<Tuple>(tuple), adjoint_expr_map_[binding->var]);
  }

  // for TupleGetItem nodes, we do a partial update
  // e.g.
  // b = a[0]
  // a_adjoint_expr[0] (in fields) += b_adjoint_var
  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* tuple_get_item) override {
    UpdateExprMap(GetRef<TupleGetItem>(tuple_get_item), adjoint_expr_map_[binding->var]);
  }

  // for assign nodes, we add the adjoint of output to the adjoint of input
  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) override {
    UpdateExprMap(GetRef<Var>(var), adjoint_var_map_[binding->var]);
  }

 private:
  Expr ReplaceExprByVar(Expr expr) {
    if (adjoint_expr_to_var_.count(expr)) {
      return adjoint_expr_to_var_[expr];
    }
    return expr;
  }

  Var CreateAdjointVar(Var v, bool is_dataflow_var) {
    // the adjoint var has been created
    if (adjoint_var_map_.count(v)) return; //?
    Var adjoint;
    if (is_dataflow_var) {
      adjoint = DataflowVar(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
    } else {
      adjoint = Var(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
    }
    adjoint_var_map_.Set(v, adjoint);
    return adjoint;
  }

  // base could only be one of these types:
  // Var, Tuple, TupleGetItem
  void UpdateExprMap(const Expr& base, const Expr& increment) {
    if (const auto* node = base.as<VarNode>()) {
      const Var& v = GetRef<Var>(node);
      if (adjoint_expr_map_.count(v) == 0) {
        adjoint_expr_map_.Set(v, ReplaceExprByVar(increment));
      } else {
        const Expr& updated = DoAdd(adjoint_expr_map_[v], increment);
        adjoint_expr_map_.Set(v, updated);
      }
    } else if (const auto* node = base.as<TupleNode>()) {
      if (const auto* node1 = increment.as<TupleNode>()) {
        for (size_t i = 0; i < node->fields.size(); ++i) {
          UpdateExprMap(node->fields[i], node1->fields[i]);
        }
      } else {
        LOG(FATAL) << "base and increment should be both tuple";
      }
    } else if (const auto* node = base.as<TupleGetItemNode>()) {
      ICHECK(node->tuple->IsInstance<VarNode>()) << "Tuple of TupleGetItem must be bound to a Var";
      ICHECK(!node->tuple->shape().as<TupleGetItemNode>()) << "Error: no nested TupleGetItem";
      ICHECK(node->tuple->shape().as<TupleNode>()) << "Type of tuple of TupleGetItem must be tuple";

      const Var& v = Downcast<Var>(node->tuple);
      if (adjoint_expr_map_.count(v) == 0) {
        const Tuple& init =
            BuildEmptyNestedTupleExpr(Downcast<Tuple>(node->tuple->shape()),
                                      Downcast<TupleType>(node->tuple->checked_type()));
        init->checked_type_ = v->checked_type();
        adjoint_expr_map_.Set(v, init);
      }

      ICHECK(adjoint_expr_map_[v].as<TupleNode>()) << "adjoint of var is not tuple";
      adjoint_expr_map_.Set(
          v, DoAddInTuple(Downcast<Tuple>(adjoint_expr_map_[v]), node->index, increment));
    } else {
      LOG(FATAL) << "not a leaf node";
    }
  }

  Tuple BuildEmptyNestedTupleExpr(const Tuple& shape, const TupleType& type) {
    static const Op& zeros_op = Op::Get("relax.zeros");

    Array<Expr> ret;
    for (size_t i = 0; i < shape->fields.size(); ++i) {
      if (const auto* node = shape->fields[i].as<TupleNode>()) {
        ret.push_back(
            BuildEmptyNestedTupleExpr(GetRef<Tuple>(node), Downcast<TupleType>(type->fields[i])));
      } else if (shape->fields[i].as<ShapeExprNode>()) {
        ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
        auto tensortype = Downcast<DynTensorType>(type->fields[i]);
        attrs->dtype = tensortype->dtype;

        const Expr& init = Call(zeros_op, {shape->fields[i]}, Attrs(attrs));
        zeros_tracker_.emplace(init);
        ret.push_back(init);
      } else {
        LOG(FATAL) << "Unsupported emtpy expr: " << shape->fields[i];
      }
    }
    return Tuple(ret);
  }

  Expr DoAdd(const Expr& src1, const Expr& src2) {
    static const Op& add_op = Op::Get("relax.add");

    if (zeros_tracker_.count(src1) != 0) {
      return ReplaceExprByVar(src2);
    } else if (zeros_tracker_.count(src2) != 0) {
      return ReplaceExprByVar(src1);
    }

    if (const auto* node1 = src1.as<TupleNode>()) {
      if (const auto* node2 = src2.as<TupleNode>()) {
        ICHECK(node1->fields.size() == node2->fields.size()) << "size of tuple not match";
        Array<Expr> result;
        for (size_t i = 0; i < node1->fields.size(); ++i) {
          result.push_back(DoAdd(node1->fields[i], node2->fields[i]));
        }
        return Tuple(result);
      } else {
        LOG(FATAL) << "Type not match: src1 and src2 should be both tuple";
        return Expr();
      }
    } else {
      return Call(add_op, {src1, ReplaceExprByVar(src2)});
    }
  }

  Tuple DoAddInTuple(const Tuple& origin, int index, const Expr& increment) {
    Array<Expr> ret;
    for (size_t i = 0; i < origin->fields.size(); ++i) {
      if ((int)i == index) {
        ret.push_back(DoAdd(origin->fields[i], increment));
      } else {
        ret.push_back(origin->fields[i]);
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

  // check that the target should be a VarNode, not DataflowVarNode
  // and a scalar of type "DynTensorType"
  void CheckTarget(Expr e) {
    CHECK(e->IsInstance<VarNode>()) << "target must be VarNode";
    CHECK(!e->IsInstance<DataflowVarNode>()) << "target is not an output node";
    CHECK(e->checked_type().as<DynTensorTypeNode>()) << "the type of target must be DynTensorType";
    CHECK(e->shape().as<ShapeExprNode>()) << "error when getting target shape";
    const auto* shape_node = e->shape().as<ShapeExprNode>();
    CHECK(shape_node->values.size() == 0) << "target must be a scalar";
  }

  // init the gradient of the target_var_
  // and update it in adjoint_expr_map_
  void InitGradAsOnes(const Var& var) {
    Expr ones = MakeOnes(var->shape(), Downcast<DynTensorType>(var->checked_type())->dtype);
    adjoint_expr_map_.Set(var, ones);
  }

  // the differentiation target
  Var target_var_;
  // the arguments to differentiate
  Array<Var> require_grads_;

  Expr return_expr_;

  // var to its adjoints var
  Map<Var, Var> adjoint_var_map_;
  // var to its adjoint expr
  Map<Var, Expr> adjoint_expr_map_;
  // trace binding
  Map<Expr, Var> adjoint_expr_to_var_;
  // track zeros introduced
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> zeros_tracker_;
};

/* This is the internal function of tvm::relax::transform::Gradient. */
IRModule Gradient(IRModule m, const GlobalVar& gvar,
                  Optional<Array<Var>> require_grads = runtime::NullOptType()) {
  auto* func = m->Lookup(gvar).as<FunctionNode>();

  if (func == nullptr) {
    LOG(FATAL) << "relax function " << gvar->name_hint << " not found";
    return m;
  }

  auto f_before = GetRef<Function>(func);
  if (require_grads.defined()) {
    for (auto input : require_grads.value()) {
      ICHECK(std::find(func->params.begin(), func->params.end(), input) != func->params.end())
          << "function " << gvar->name_hint << " has no var named " << input->name_hint();
    }
  } else {
    // the default case, require_grads = all params of function
    require_grads = func->params;
  }

  IRModuleNode* new_module_node = m.CopyOnWrite();
  auto new_module = GetRef<IRModule>(new_module_node);
  auto mutator = GradientMutator(new_module, require_grads.value());

  auto adjoint_var = GlobalVar(gvar->name_hint + "_adjoint");
  Function f_after = mutator.FuncTransform(GetRef<Function>(func));
  new_module->Add(adjoint_var, f_after);

  return new_module;
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
