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
  explicit SimpleADMutator(IRModule mod, const Array<Var>& require_grads)
      : ExprMutator(mod), require_grads(require_grads) {}

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

    // create adjoint var for inputs
    for (const auto& v: node->params) {
      if (require_grads.empty() || CheckArrayContains(require_grads, v)) {
        CreateAdjointVar(v, false);
      }
      else {
        CreateAdjointVar(v, true);
      }
    }

    // the return value of the function will be treated as target
    if (const auto* node = seq_expr->body.as<VarNode>()) {
      const Var& target = GetRef<Var>(node);
      CheckTarget(target);
      CreateAdjointVar(target, true);
      InitGrad(adjoint_var_map[target], target);
    }
    else {
      LOG(FATAL) << "the body of the function (the default target) is not a relax.Var";
    }

    // reverse-mode ad
    for (int i = block->bindings.size()-1; i >= 0; --i) {
      if (!block->bindings[i]->IsInstance<VarBindingNode>()) continue;
      const VarBindingNode* binding = block->bindings[i].as<VarBindingNode>();
      VisitBinding_(binding);
    }

    // handle the return values and types
    Array<Expr> out_expr, out_adjoints;
    Array<Type> ret_type, out_adjoints_type;
    out_expr.push_back(seq_expr->body);
    ret_type.push_back(node->ret_type);

    // emit the input adjoints
    for (const auto& param: node->params) {
      if (require_grads.empty() || CheckArrayContains(require_grads, param)) {
        const Var& adjoint_var = adjoint_var_map[param];
        if (adjoint_expr_map.count(param)) {
          BindAndEmit(adjoint_var, adjoint_expr_map[param]);
        }
        else {
          const Expr& default_adjoint = Call(default_op, {param->shape()});
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
    VLOG(2) << "[AD] Visit Binding: " << binding->var->name_hint() << std::endl;
    CreateAdjointVar(binding->var, true);
    const Var& adjoint_var = adjoint_var_map[binding->var];

    // must be ignored output's AST
    if (adjoint_expr_map.count(binding->var) == 0) {
      VLOG(2) << "ignored: " << binding->var->name_hint() << std::endl;
      return;
    }
    
    // meet a def
    BindAndEmit(adjoint_var, adjoint_expr_map[binding->var]);
    // back prop.
    ICHECK(adjoint_expr_map.count(binding->var)) << "AD error: lhs has no adjoint" << std::endl;

    // case 1: tuple
    // a = ((c, d),)
    // b_adjoint_expr += a_adjoint_var[0], c_adjoint_expr += a_adjoint_var[1]
    if (const auto* node = binding->value.as<TupleNode>()) {
      UpdateExprMap(GetRef<Tuple>(node), adjoint_expr_map[binding->var]);
    }
    // case 2: tuple get item
    // b = a[0]
    // a_adjoint_expr[0] (in fields) += b_adjoint_var
    // a = ((x, y), (z,))
    // b = a[0]
    else if (const auto* node = binding->value.as<TupleGetItemNode>()) {
      UpdateExprMap(GetRef<TupleGetItem>(node), adjoint_expr_map[binding->var]);
    }
    // case 3: assign
    // a = b
    // b_adjoint_expr += a_adjoint_var
    else if (const auto* node = binding->value.as<VarNode>()) {
      UpdateExprMap(GetRef<Var>(node), adjoint_expr_map[binding->var]);
    }
    // case 4: call
    else if (const auto* node = binding->value.as<CallNode>()) {
      const Op& call_op = GetRef<Op>(node->op.as<OpNode>());
      const Array<Expr>& partials = gradient_op_map[call_op](GetRef<Call>(node), adjoint_var);
      ICHECK(partials.size() == node->args.size()) << "partials number != inputs number";
      for (size_t i = 0; i < partials.size(); ++i) {
        const VarNode* arg = node->args[i].as<VarNode>();
        ICHECK(arg != nullptr);
        UpdateExprMap(GetRef<Var>(arg), partials[i]);
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
  template<typename T>
  static bool CheckArrayContains(const Array<T> array, T value) {
    for (auto i : array) {
      if (i == value) {
        return true;
      }
    }
    return false;
  }


  void CreateAdjointVar(const Var& v, bool is_dataflow_var) {
    // the adjoint var has been created
    if (adjoint_var_map.count(v)) return;
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

  void UpdateExprMap(const Expr& base, const Expr& increment) {
    if (const auto* node = base.as<VarNode>()) {
      const Var& v = GetRef<Var>(node);
      if (adjoint_expr_map.count(v) == 0) {
        if (adjoint_binding_.count(increment)) {
          adjoint_expr_map.Set(v, adjoint_binding_[increment]);
        }
        else {
          adjoint_expr_map.Set(v, increment);  
        }
      }
      else {
        const Expr& updated = DoAdd(adjoint_expr_map[v], increment); // Call(add_op, {adjoint_expr_map[v], increment});
        adjoint_expr_map.Set(v, updated);
      }
    }
    else if (const auto* node = base.as<TupleNode>()) {
      if (const auto* node1 = increment.as<TupleNode>()) {
        for (size_t i = 0; i < node->fields.size(); ++i) {
          UpdateExprMap(node->fields[i], node1->fields[i]);
        }
      }
      else {
        LOG(FATAL) << "Type not match: base and increment should be both tuple" << std::endl;
      }
    }
    else if (const auto* node = base.as<TupleGetItemNode>()) {
      ICHECK(node->tuple->IsInstance<VarNode>()) << "Tuple of TupleGetItem must be binded to a Var" << std::endl;
      ICHECK(!node->tuple->shape().as<TupleGetItemNode>()) << "Error: no nested TupleGetItem" << std::endl;
      ICHECK(node->tuple->shape().as<TupleNode>()) << "Type of tuple of TupleGetItem must be tuple" << std::endl;
      
      const Var& v = GetRef<Var>(node->tuple.as<VarNode>());
      if (adjoint_expr_map.count(v) == 0) {
        const Tuple& init = BuildEmptyNestedTupleExpr(GetRef<Tuple>(node->tuple->shape().as<TupleNode>()));
        init->checked_type_ = v->checked_type();
        adjoint_expr_map.Set(v, init);
      }
      
      ICHECK(adjoint_expr_map[v].as<TupleNode>()) << "adjoint of var is not tuple";
      adjoint_expr_map.Set(v, 
        DoAddInTuple(GetRef<Tuple>(adjoint_expr_map[v].as<TupleNode>()), node->index, increment)
      );
    }
    else {
      LOG(FATAL) << "not a leaf node" << std::endl;
    }
  }

  Tuple BuildEmptyNestedTupleExpr(const Tuple& from) {
    Array<Expr> ret;
    for (size_t i = 0; i < from->fields.size(); ++i) {
      if (const auto* node = from->fields[i].as<TupleNode>()) {
        ret.push_back(BuildEmptyNestedTupleExpr(GetRef<Tuple>(node)));
      }
      else if (from->fields[i].as<ShapeExprNode>()) {
        // VLOG(2) << "init: " << from << std::endl;
        const Expr& init = Call(default_op, {from->fields[i]});
        zeros_tracker_.emplace(init);
        ret.push_back(init);
      }
      else {
        LOG(FATAL) << "Unsupported emtpy expr: " << from->fields[i] << std::endl;
      }
    }
    return Tuple(ret);
  }

  Expr DoAdd(const Expr& src1, const Expr& src2) {
    VLOG(2) << "DoAdd." << std::endl;
    VLOG(2) << "src1: " << src1 << std::endl; 
    VLOG(2) << "src2: " << src2 << std::endl; 
    
    if (zeros_tracker_.count(src1) != 0) {
      return src2;
    }
    else if (zeros_tracker_.count(src2) != 0) {
      return src1;
    }

    if (const auto* node1 = src1.as<TupleNode>()) {
      if (const auto* node2 = src2.as<TupleNode>()) {
        ICHECK(node1->fields.size() == node2->fields.size()) << "size of tuple not match" << std::endl;
        Array<Expr> result;
        for (size_t i = 0; i < node1->fields.size(); ++i) {
          result.push_back(DoAdd(node1->fields[i], node2->fields[i]));
        }
        return Tuple(result);
      }
      else {
        LOG(FATAL) << "Type not match: src1 and src2 should be both tuple" << std::endl;
        return Expr();
      }
    }
    else {
      // use the variable to replace expr to reduce the size of AST
      if (adjoint_binding_.count(src2)) {
        return Call(add_op, {src1, adjoint_binding_[src2]});
      } 
      return Call(add_op, {src1, src2});
    }
  }

  Tuple DoAddInTuple(const Tuple& origin, int index, const Expr& increment) {
    Array<Expr> ret;
    for (size_t i = 0; i < origin->fields.size(); ++i) {
      if ((int)i == index) {
        ret.push_back(DoAdd(origin->fields[i], increment));
      }
      else {
        ret.push_back(origin->fields[i]);
      }
    }
    return Tuple(ret);
  }

  void BindAndEmit(const Var& v, const Expr& e) {
    if (adjoint_binding_.count(e)) {
      return;
    }
    adjoint_binding_.Set(e, v);
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
    ICHECK(e->checked_type_.as<DynTensorTypeNode>()) << "target must be a DynTensorType";
    ICHECK(e->shape().as<ShapeExprNode>()) << "error when getting target shape";
    const auto* shape_node = e->shape().as<ShapeExprNode>();
    ICHECK(shape_node->values.size() == 0) << "target must be a scalar";
  }

  void InitGrad(const Var& adjoint_var, const Var& var) {
    const Expr& init = Call(init_op, {var});
    BindAndEmit(adjoint_var, init);
    VLOG(2) << "init grad: " << var->name_hint() << std::endl;
    adjoint_expr_map.Set(var, init);
  }

  Array<Var> require_grads;

  // var to its adjoints var
  Map<Var, Var> adjoint_var_map;
  // var to its adjoint expr
  Map<Var, Expr> adjoint_expr_map;

  // trace binding
  Map<Expr, Var> adjoint_binding_;
  // track zeros introduced
  std::set<Expr> zeros_tracker_;

  // gop map
  const OpAttrMap<relay::FPrimalGradient> gradient_op_map =
      Op::GetAttrMap<relay::FPrimalGradient>("FPrimalGradient");
  
  // constant
  const Op& init_op = Op::Get("relax.ones_like");
  const Op& add_op = Op::Get("relax.add");
  const Op& default_op = Op::Get("relax.zeros");
};

/*!
 * \brief A simple reverse-mode auto differentiation.
 * \param m The module
 * \param func_name The name of the specific function
 * \param require_grad_names The relax variables which need adjoints. Must be inputs.
 * \return The module after AD.
 */
IRModule SimpleAD(IRModule m, const GlobalVar &var, const Array<ObjectRef> &require_grads) {
  BaseFunc base_func = m->Lookup(var);
  if (auto* n = base_func.as<FunctionNode>()) {
    auto f_before = GetRef<Function>(n);
    Array<Var> require_grads_var;
    for (auto input : require_grads) {
      if (auto* n = input.as<IntImmNode>()) {
        int64_t idx = GetRef<Integer>(n).IntValue();
        require_grads_var.push_back(f_before->params[idx]);
      } else if (auto *n = input.as<VarNode>()){
        require_grads_var.push_back(GetRef<Var>(n));
      } else {
        LOG(FATAL) << "require_grads argument of the SimpleAD call has wrong type";
      }
    }

    IRModuleNode* new_module_node = m.CopyOnWrite();
    auto new_module = GetRef<IRModule>(new_module_node);
    auto mutator = SimpleADMutator(new_module, require_grads_var);

    auto adjoint_var = GlobalVar(var->name_hint + "_adjoint");
    Function f_after = Downcast<Function>(mutator.VisitExpr(f_before));
    f_after = WithAttr(f_after, tvm::attr::kGlobalSymbol, adjoint_var->name_hint);
    new_module->Add(adjoint_var, f_after);

    return new_module;
  } else {
    LOG(FATAL) << "Relax function " << var->name_hint << " not found";
    return m;
  }
}

namespace transform {

Pass SimpleAD(GlobalVar func, Array<ObjectRef> require_grads) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) {
        return relax::SimpleAD(mod, func, require_grads);
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
