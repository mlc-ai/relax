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
 * \file src/relax/transform/autodiff/simple_ad.cc
 * \brief A simple reverse-mode auto differentiation.
 *
 * Now only supports differentiating a function in the IRModule with one dataflow block
 * with respect to the only return value of the function. It needs to be scalar.
 *
 * For examples, see tests/python/relax/test_transform_autodiff.py.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/op_attr_types.h>

#include <unordered_set>

namespace tvm {
namespace relax {

class GradientMutator : public ExprMutator {
 public:
  explicit GradientMutator(IRModule mod, const Array<Var>& require_grads)
      : ExprMutator(mod), require_grads_(require_grads) {}

  Function FuncTransform(const FunctionNode* node) {
    ICHECK(node->body->IsInstance<SeqExprNode>());

    Array<Var> new_params;
    for (Var param : node->params) {
      Var new_param = Var(param->vid, param->shape(), param->checked_type(), param->span);
      this->var_remap_[param->vid] = new_param;
      new_params.push_back(new_param);
    }
    Expr new_body = VisitWithNewScope(node->body);

    const SeqExprNode* seq_expr = new_body.as<SeqExprNode>();
    // only support a single dataflow block
    ICHECK(seq_expr->blocks.size() == 1);
    ICHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockNode>());
    const DataflowBlockNode* block = seq_expr->blocks[0].as<DataflowBlockNode>();

    builder_->BeginDataflowBlock();
    for (const auto& binding: block->bindings) {
      VisitBinding(binding);
    }

    // create adjoint var for inputs
    for (size_t i = 0; i < new_params.size(); ++i) {
      if (require_grads_.empty() ||
        std::find(require_grads_.begin(), require_grads_.end(), node->params[i]) != require_grads_.end()) {
        CreateAdjointVar(new_params[i], false);
      } else {
        CreateAdjointVar(new_params[i], true);
      }
    }

    // the return value of the function will be treated as target
    if (const auto* node = seq_expr->body.as<VarNode>()) {
      const Var& target = GetRef<Var>(node);
      CheckTarget(target);
      target_var_ = target;
    } else {
      LOG(FATAL) << "the body of the function is not a relax.Var";
    }

    // reverse-mode ad
    for (int i = block->bindings.size() - 1; i >= 0; --i) {
      if (!block->bindings[i]->IsInstance<VarBindingNode>()) continue;
      const VarBindingNode* binding = block->bindings[i].as<VarBindingNode>();
      ReverseVisit(binding);
    }

    // handle the return values and types
    Array<Expr> out_expr, out_adjoints, out_shape, out_adjoints_shape;
    Array<Type> ret_type, out_adjoints_type;
    out_expr.push_back(seq_expr->body);
    out_shape.push_back(seq_expr->body->shape());
    ret_type.push_back(node->ret_type);

    // emit the input adjoints
    for (size_t i = 0; i < new_params.size(); ++i) {
      if (require_grads_.empty() ||
        std::find(require_grads_.begin(), require_grads_.end(), node->params[i]) != require_grads_.end()) {
        const Var& adjoint_var = adjoint_var_map_[new_params[i]];
        if (adjoint_expr_map_.count(new_params[i])) {
          BindAndEmit(adjoint_var, adjoint_expr_map_[new_params[i]]);
        } else {
          ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
          auto type = Downcast<DynTensorType>(new_params[i]->checked_type());
          attrs->dtype = type->dtype;

          const Expr& default_adjoint = Call(zeros_op, {new_params[i]->shape()}, Attrs(attrs));
          BindAndEmit(adjoint_var, default_adjoint);
        }
        out_adjoints.push_back(adjoint_var);
        out_adjoints_type.push_back(adjoint_var->checked_type());
        out_adjoints_shape.push_back(adjoint_var->shape());
      }
    }

    out_expr.push_back(Tuple(out_adjoints));
    out_shape.push_back(Tuple(out_adjoints_shape));
    ret_type.push_back(TupleType(out_adjoints_type));

    Type new_ret_type = VisitType(TupleType(ret_type));
    Expr final_body = builder_->Normalize(SeqExpr({builder_->EndBlock()}, Tuple(out_expr)));

    return Function(new_params, final_body, new_ret_type, /*Tuple(out_shape)*/ RuntimeDepShape(), node->attrs);
  }

  void ReverseVisit(const VarBindingNode* binding) {
    CreateAdjointVar(binding->var, true);
    const Var& adjoint_var = adjoint_var_map_[binding->var];

    // must be ignored output's AST
    if (adjoint_expr_map_.count(binding->var) == 0) {
      if (binding->var == target_var_) {
        InitGrad(adjoint_var_map_[binding->var], binding->var);
      } else {
        return;
      }
    }

    // meet a def
    BindAndEmit(adjoint_var, adjoint_expr_map_[binding->var]);

    // back propagation
    if (const auto* node = binding->value.as<TupleNode>()) {
      // case 1: tuple
      // a = ((c, d),)
      // b_adjoint_expr += a_adjoint_var[0], c_adjoint_expr += a_adjoint_var[1]
      UpdateExprMap(GetRef<Tuple>(node), adjoint_expr_map_[binding->var]);
    } else if (const auto* node = binding->value.as<TupleGetItemNode>()) {
      // case 2: tuple get item
      // b = a[0]
      // a_adjoint_expr[0] (in fields) += b_adjoint_var
      // a = ((x, y), (z,))
      // b = a[0]
      UpdateExprMap(GetRef<TupleGetItem>(node), adjoint_expr_map_[binding->var]);
    } else if (const auto* node = binding->value.as<VarNode>()) {
      // case 3: assign
      // a = b
      // b_adjoint_expr += a_adjoint_var
      UpdateExprMap(GetRef<Var>(node), adjoint_expr_map_[binding->var]);
    } else if (const auto* node = binding->value.as<CallNode>()) {
      // case 4: call
      const Op& call_op = GetRef<Op>(node->op.as<OpNode>());
      const Array<Expr>& partials = gradient_op_map[call_op](GetRef<Call>(node), adjoint_var);
      ICHECK(partials.size() == node->args.size()) << "partials number != inputs number";
      for (size_t i = 0; i < partials.size(); ++i) {
        const VarNode* arg = node->args[i].as<VarNode>();
        ICHECK(arg != nullptr);
        UpdateExprMap(GetRef<Var>(arg), partials[i]);
      }
    } else {
      LOG(FATAL) << "AD does not support this type of binding value now: " << binding->value;
    }
  }

 private:
  Expr ReplaceExprByVar(Expr expr) {
    if (adjoint_expr_to_var_.count(expr)) {
      return adjoint_expr_to_var_[expr];
    }
    return expr;
  }

  void CreateAdjointVar(const Var& v, bool is_dataflow_var) {
    // the adjoint var has been created
    if (adjoint_var_map_.count(v)) return;
    if (is_dataflow_var) {
      Var adjoint = DataflowVar(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
      adjoint_var_map_.Set(v, adjoint);
    } else {
      Var adjoint = Var(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
      adjoint_var_map_.Set(v, adjoint);
    }
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
    }
    else if (const auto* node = base.as<TupleNode>()) {
      if (const auto* node1 = increment.as<TupleNode>()) {
        for (size_t i = 0; i < node->fields.size(); ++i) {
          UpdateExprMap(node->fields[i], node1->fields[i]);
        }
      } else {
        LOG(FATAL) << "base and increment should be both tuple";
      }
    }
    else if (const auto* node = base.as<TupleGetItemNode>()) {
      ICHECK(node->tuple->IsInstance<VarNode>()) << "Tuple of TupleGetItem must be binded to a Var";
      ICHECK(!node->tuple->shape().as<TupleGetItemNode>()) << "Error: no nested TupleGetItem";
      ICHECK(node->tuple->shape().as<TupleNode>()) << "Type of tuple of TupleGetItem must be tuple";

      const Var& v = Downcast<Var>(node->tuple);
      if (adjoint_expr_map_.count(v) == 0) {
        const Tuple& init = BuildEmptyNestedTupleExpr(
          Downcast<Tuple>(node->tuple->shape()),
          Downcast<TupleType>(node->tuple->checked_type())
        );
        init->checked_type_ = v->checked_type();
        adjoint_expr_map_.Set(v, init);
      }

      ICHECK(adjoint_expr_map_[v].as<TupleNode>()) << "adjoint of var is not tuple";
      adjoint_expr_map_.Set(v,
        DoAddInTuple(Downcast<Tuple>(adjoint_expr_map_[v]), node->index, increment)
      );
    } else {
      LOG(FATAL) << "not a leaf node";
    }
  }

  Tuple BuildEmptyNestedTupleExpr(const Tuple& shape, const TupleType &type) {
    Array<Expr> ret;
    for (size_t i = 0; i < shape->fields.size(); ++i) {
      if (const auto* node = shape->fields[i].as<TupleNode>()) {
        ret.push_back(BuildEmptyNestedTupleExpr(
          GetRef<Tuple>(node), Downcast<TupleType>(type->fields[i])
        ));
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
    if (zeros_tracker_.count(src1) != 0) {
      return ReplaceExprByVar(src2);
    }
    else if (zeros_tracker_.count(src2) != 0) {
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
      adjoint_expr_to_var_.Set(e, v);
      // e->checked_type_ = v->checked_type();
      // e->shape_ = v->shape();
    }
    if (v->IsInstance<DataflowVarNode>()) {
      builder_->Emit(VarBinding(v, e));
    } else {
      builder_->EmitOutput(VarBinding(v, e));
    }
  }

  void CheckTarget(const Expr& e) {
    ICHECK(!e->IsInstance<DataflowVarNode>()) << "not an output node";
    ICHECK(e->checked_type().as<DynTensorTypeNode>()) << "target must be a DynTensorType";
    ICHECK(e->shape().as<ShapeExprNode>()) << "error when getting target shape";
    const auto* shape_node = e->shape().as<ShapeExprNode>();
    ICHECK(shape_node->values.size() == 0) << "target must be a scalar";
  }

  void InitGrad(const Var& adjoint_var, const Var& var) {
    ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
    auto type = Downcast<DynTensorType>(var->checked_type());
    attrs->dtype = type->dtype;

    const Expr& init = Call(ones_op, {var->shape()}, Attrs(attrs));
    adjoint_expr_map_.Set(var, init);
  }

  // the differentiation target
  Var target_var_;
  // the arguments to differentiate
  Array<Var> require_grads_;

  // var to its adjoints var
  Map<Var, Var> adjoint_var_map_;
  // var to its adjoint expr
  Map<Var, Expr> adjoint_expr_map_;
  // trace binding
  Map<Expr, Var> adjoint_expr_to_var_;
  // track zeros introduced
  std::set<Expr> zeros_tracker_;

  // gop map
  const OpAttrMap<FPrimalGradient> gradient_op_map =
      Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");

  // constant
  const Op& ones_op = Op::Get("relax.ones");
  const Op& add_op = Op::Get("relax.add");
  const Op& zeros_op = Op::Get("relax.zeros");
};

/*!
 * \brief A simple reverse-mode auto differentiation.
 * \param m The module
 * \param var The GlobalVar of the specific function
 * \param require_grad_names The relax variables which need adjoints. Must be inputs.
 * \return The module after AD.
 */
IRModule Gradient(IRModule m, const GlobalVar &var, const Array<Var> &require_grads) {
  BaseFunc base_func = m->Lookup(var);
  if (auto* n = base_func.as<FunctionNode>()) {
    auto f_before = GetRef<Function>(n);
    Array<Var> require_grads_var;
    for (auto input: require_grads) {
      ICHECK(std::find(n->params.begin(), n->params.end(), input) != n->params.end())
      << "function " << var->name_hint << " has no var named " << input->name_hint();
      require_grads_var.push_back(input);
    }

    IRModuleNode* new_module_node = m.CopyOnWrite();
    auto new_module = GetRef<IRModule>(new_module_node);
    auto mutator = GradientMutator(new_module, require_grads_var);

    auto adjoint_var = GlobalVar(var->name_hint + "_adjoint");
    Function f_after = mutator.FuncTransform(f_before.as<FunctionNode>());
    f_after = WithAttr(f_after, tvm::attr::kGlobalSymbol, adjoint_var->name_hint);
    new_module->Add(adjoint_var, f_after);

    return new_module;
  } else {
    LOG(FATAL) << "relax function " << var->name_hint << " not found";
    return m;
  }
}

namespace transform {

Pass Gradient(GlobalVar func, Array<Var> require_grads) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) {
        return relax::Gradient(mod, func, require_grads);
      };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"Gradient",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.Gradient").set_body_typed(Gradient);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
