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
 * with respect to one of its output(s). The specified output needs to be scalar.
 *
 * Example:
 *
 * Before AD:
 * @tvm.script.ir_module
 * class Before:
 *     @R.function
 *     def main(x: Tensor((5, 5), "float32"),
 *              y: Tensor((5, 5), "float32")):
 *         with R.dataflow():
 *             lv0 = relax.add(x, y)
 *             lv1 = relax.sum(lv0)
 *             R.output(lv1)
 *         return lv1
 *
 * After AD:
 * @tvm.script.ir_module
 * class Module:
 *     @R.function
 *     def main(x: Tensor((5, 5), "float32"), y: Tensor((5, 5), "float32")) -> Tuple(
 *             Tensor(None, "float32", ndim = 0), Tuple(Tensor(None, "float32", ndim = 2),
 *             Tensor(None, "float32", ndim = 2))):
 *         # block 0
 *         with R.dataflow():
 *             lv0: Tensor((5, 5), "float32") = relax.add(x, y)
 *             lv1: Tensor((), "float32") = relax.sum(lv0)
 *             lv1_adjoint: Tensor((), "float32") = relax.ones_like(lv1)
 *             lv: Tensor((5, 5), "float32") = relax.ones_like(lv0)
 *             lv0_adjoint: Tensor((5, 5), "float32") = relax.multiply(lv1_adjoint, lv)
 *             x_adjoint: Tensor((5, 5), "float32") = relax.collapse_sum_like(lv0_adjoint, x)
 *             y_adjoint: Tensor((5, 5), "float32") = relax.collapse_sum_like(lv0_adjoint, y)
 *             R.output(lv1, x_adjoint, y_adjoint)
 *         # return value type: Tuple(original_return_value, Tuple(all_adjoints))
 *         return (lv1, (x_adjoint, y_adjoint))
 *
 *  TODO(yixindong, chaofanlin): eliminate unnecessary computations.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/op_attr_types.h>

#include <unordered_set>

namespace tvm {
namespace relax {

class SimpleADMutator : public ExprMutator {
 public:
  explicit SimpleADMutator(IRModule mod, const String& target_name,
                           const Array<String>& require_grad_names)
      : ExprMutator(mod), target_name_(target_name), require_grad_names_() {
    for (const String& name: require_grad_names) {
      require_grad_names_.emplace(name);
    }
  }

  Expr VisitExpr_(const FunctionNode* node) override {
    ICHECK(node->body->IsInstance<SeqExprNode>());
    const SeqExprNode* seq_expr = node->body.as<SeqExprNode>();
    // only a single dataflow block
    ICHECK(seq_expr->blocks.size() == 1);
    ICHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockNode>());
    const DataflowBlockNode* block = seq_expr->blocks[0].as<DataflowBlockNode>();

    builder_->BeginDataflowBlock();
    // copy and emit
    for (const auto& binding: block->bindings) {
      EmitBinding(binding);
    }

    for (const auto& v: node->params) {
      if (require_grad_names_.empty() || require_grad_names_.count(v->name_hint())) {
        CreateAdjointVar(v, false);
      }
      else {
        CreateAdjointVar(v, true);
      }
    }

    // if target is not specified
    if (target_name_.empty()) {
      if (const auto* node = seq_expr->body.as<VarNode>()) {
        const Var& body_var = GetRef<Var>(node);
        CheckTarget(body_var);
        CreateAdjointVar(body_var, true);
        InitGrad(adjoint_var_map[body_var], body_var);
      }
      else {
        LOG(FATAL) << "the body of the function (the default target) is not a relax.Var" << std::endl;
      }
    }

    // reverse-mode
    for (int i = block->bindings.size()-1; i >= 0; --i) {
      if (!block->bindings[i]->IsInstance<VarBindingNode>()) continue;
      const VarBindingNode* binding = block->bindings[i].as<VarBindingNode>();
      VisitBinding_(binding);
    }

    // handle the return
    Array<Expr> out_expr, out_adjoints;
    Array<Type> ret_type, out_adjoints_type;
    out_expr.push_back(seq_expr->body);
    ret_type.push_back(node->ret_type);

    // emit the input adjoints
    for (const auto& param: node->params) {
      if (require_grad_names_.empty() || require_grad_names_.count(param->name_hint())) {
        const Var& adjoint_var = adjoint_var_map[param];
        if (adjoint_expr_map.count(param)) {
          BindAndEmit(adjoint_var, adjoint_expr_map[param]);
        }
        else {
          const Op& op = Op::Get("relax.zeros_like");
          const Expr& default_adjoint = Call(op, {param});
          BindAndEmit(adjoint_var, default_adjoint);
        }
        out_adjoints.push_back(adjoint_var);
        out_adjoints_type.push_back(adjoint_var->checked_type());
      }
    }

    out_expr.push_back(Tuple(out_adjoints));
    ret_type.push_back(TupleType(out_adjoints_type));

    return Function(node->params, SeqExpr({builder_->EndBlock()}, Tuple(out_expr)),
                    TupleType(ret_type), node->attrs);
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    CreateAdjointVar(binding->var, true);
    const Var& adjoint_var = adjoint_var_map[binding->var];

    // must be output or expr in ignored output's AST
    if (adjoint_expr_map.count(binding->var) == 0) {
      if (target_name_.empty() || target_name_ != binding->var->name_hint()) {
        return;
      }
      // if target is specified
      CheckTarget(binding->var);
      InitGrad(adjoint_var, binding->var);
    }
    else {
      // meet a def
      BindAndEmit(adjoint_var, adjoint_expr_map[binding->var]);
    }

    // back prop.

    // case 1: assign
    // a = b
    // b_adjoint_expr += a_adjoint_var
    if (const auto* node = binding->value.as<VarNode>()) {
      AdjointExprIncre(GetRef<Var>(node), adjoint_var);
    }
    // case 2: call
    else if (const auto* node = binding->value.as<CallNode>()) {
      const Op& call_op = GetRef<Op>(node->op.as<OpNode>());
      const Array<Expr>& partials = gradient_op_map[call_op](GetRef<Call>(node), adjoint_var);
      ICHECK(partials.size() == node->args.size()) << "partials number != inputs number";
      for (size_t i = 0; i < partials.size(); ++i) {
        const VarNode* arg = node->args[i].as<VarNode>();
        ICHECK(arg != nullptr);
        AdjointExprIncre(GetRef<Var>(arg), partials[i]);
      }
    }
    else {
      LOG(FATAL) << "Unsupport: unknown binding expr" << binding->value;
    }

    // SSA. release the space
    adjoint_var_map.erase(binding->var);
    adjoint_expr_map.erase(binding->var);
  }

 private:
  void CreateAdjointVar(const Var& v, bool is_dataflow_var) {
    // the adjoint var has been created
    if (adjoint_var_map.count(v) != 0) return;
    if (is_dataflow_var) {
      Var adjoint = DataflowVar(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
      adjoint->checked_type_ = v->checked_type();
      adjoint_var_map.Set(v, adjoint);
    }
    else {
      Var adjoint = Var(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
      adjoint->checked_type_ = v->checked_type();
      adjoint_var_map.Set(v, adjoint);
    }
  }

  void AdjointExprIncre(const Var& v, const Expr& increment) {
    if (adjoint_expr_map.count(v) == 0) {
      adjoint_expr_map.Set(v, increment);
    }
    else {
      const Expr& now_expr = adjoint_expr_map[v];
      const Op& add_op = Op::Get("relax.add");
      const Expr& new_expr = Call(add_op, {now_expr, increment});
      adjoint_expr_map.Set(v, new_expr);
    }
  }

  void EmitBinding(const Binding& binding) {
    if (const auto* node = binding.as<VarBindingNode>()) {
      const VarBinding& var_binding = GetRef<VarBinding>(node);
      if (var_binding->var->IsInstance<DataflowVarNode>()) {
        builder_->Emit(var_binding);
      }
      else {
        builder_->EmitOutput(var_binding);
      }
    }
    else if (const auto* node = binding.as<MatchShapeNode>()) {
      const MatchShape& match_shape = GetRef<MatchShape>(node);
      builder_->EmitMatchShape(match_shape);
    }
    else {
      LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
    }
  }

  void BindAndEmit(const Var& v, const Expr& e) {
    e->checked_type_ = v->checked_type();
    e->shape_ = v->shape();
    if (v->IsInstance<DataflowVarNode>()) {
      builder_->Emit(VarBinding(v, e));
    }
    else {
      builder_->EmitOutput(VarBinding(v, e));
    }
  }

  void CheckTarget(const Expr& e) {
    ICHECK(!e->IsInstance<DataflowVarNode>()) << "not an output node";
    ICHECK(e->checked_type_.as<DynTensorTypeNode>()) << "target must be a DynTensorType" << std::endl;
    ICHECK(e->shape().as<ShapeExprNode>()) << "error when getting target shape" << std::endl;
    const auto* shape_node = e->shape().as<ShapeExprNode>();
    ICHECK(shape_node->values.size() == 0) << "target must be a scalar" << std::endl;
  }

  void InitGrad(const Var& adjoint_var, const Var& var) {
    const Op& init_op = Op::Get("relax.ones_like");
    BindAndEmit(adjoint_var, Call(init_op, {var}));
  }

  // specified sets
  String target_name_;
  std::unordered_set<String> require_grad_names_;

  // var to its adjoints var
  Map<Var, Var> adjoint_var_map;
  // var to its adjoint expr
  Map<Var, Expr> adjoint_expr_map;

  // gop map
  const OpAttrMap<relay::FPrimalGradient> gradient_op_map =
      Op::GetAttrMap<relay::FPrimalGradient>("FPrimalGradient");
};

/*!
 * \brief A simple reverse-mode auto differentiation.
 * \param m The module
 * \param func_name The name of the specific function
 * \param target_name The name of the relax variable that serves as target in the differentiation.
 * \param require_grad_names The relax variables which need adjoints. Must be inputs.
 * \return The module after AD.
 */
IRModule SimpleAD(IRModule m, const String& func_name, const String& target_name,
                  const Array<String>& require_grad_names) {
  IRModuleNode* new_module = m.CopyOnWrite();
  auto mutator = SimpleADMutator(GetRef<IRModule>(new_module), target_name, require_grad_names);
  bool found = false;
  for (const auto& func_pr : m->functions) {
    if (const auto* relax_f = func_pr.second.as<FunctionNode>()) {
      Optional<String> gsymbol = relax_f->GetAttr<String>(tvm::attr::kGlobalSymbol);
      if (gsymbol.defined() && gsymbol.value() == func_name) {
        Function f_after = Downcast<Function>(mutator.VisitExpr(func_pr.second));
        new_module->Update(func_pr.first, f_after);
        found = true;
      }
    }
  }
  ICHECK(found) << "function " << func_name << " not found" << std::endl;
  return GetRef<IRModule>(new_module);
}

namespace transform {

Pass SimpleAD(String func_name, String target_name, Array<String> require_grad_names) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) {
        return relax::SimpleAD(mod, func_name, target_name, require_grad_names);
      };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"SimpleAD",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.SimpleAD").set_body_typed(SimpleAD);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
