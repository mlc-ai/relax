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

#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

/*! \brief Helper to implement bind params.*/
class ExprBinder : public ExprMutator {
 public:
  explicit ExprBinder(const tvm::Map<Var, Expr>& args_map) : args_map_(args_map) {}

  Expr VisitExpr_(const VarNode* op) final {
    auto id = GetRef<Var>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

 private:
  const tvm::Map<Var, Expr>& args_map_;
};

/*!
 * \brief Bind params on expr
 * \param expr The expr where to bind params
 * \param args_map The map from param var to the expr it binds to
 * \return The result expr after bind params
 */
Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& args_map) {
  if (const FunctionNode* func = expr.as<FunctionNode>()) {
    Expr new_body = ExprBinder(args_map).VisitExpr(func->body);
    Array<Var> new_params;
    for (size_t i = 0; i < func->params.size(); ++i) {
      if (!args_map.count(func->params[i])) {
        new_params.push_back(func->params[i]);
      }
    }
    if (new_body.same_as(func->body) && new_params.size() == func->params.size()) {
      return expr;
    }
    // The checked_type_ of the new function is deduced from the function body
    // TODO(@relax-team): Should infer the shape from the body as well
    return Function(new_params, new_body, NullOpt, func->attrs);
  } else {
    return ExprBinder(args_map).VisitExpr(expr);
  }
}

bool IsBoolScalarType(const Type& ty, bool permit_unknown_rank, bool permit_unknown_dtype) {
  const DynTensorTypeNode* tt = ty.as<DynTensorTypeNode>();
  if (!tt) {
    return false;
  }
  bool correct_dtype = tt->dtype.is_bool() || (permit_unknown_dtype && tt->dtype.is_void());
  bool correct_rank = tt->ndim == 0 || (permit_unknown_rank && tt->ndim == -1);
  return correct_dtype && correct_rank;
}

bool IsLeafExpr(const Expr& expr) {
  // NB: tuples are treated as leaf nodes for ergonomics
  return expr.as<VarNode>() || expr.as<GlobalVarNode>() || expr.as<ConstantNode>() ||
         expr.as<ShapeExprNode>() || expr.as<ExternFuncNode>() || expr.as<OpNode>() ||
         expr.as<TupleNode>();
}

class FunctionCopier : public ExprMutator {
 public:
  static Function Transform(Function func) {
    FunctionCopier copier;
    // the parameters would be copied and substituted to satisfy the restriction in the well-formed
    // check: any two functions cannot share the same parameter variable.
    Array<Var> new_params;
    for (Var param : func->params) {
      Var new_param = Var(param->vid, GetStructInfo(param), param->span);
      copier.var_remap_[param->vid] = new_param;
      new_params.push_back(new_param);
    }

    Expr body = copier.VisitWithNewScope(func->body, new_params);

    return Function(new_params, body, func->ret_struct_info, func->attrs);
  }
};

Function CopyWithNewParams(Function func) { return FunctionCopier::Transform(func); }

TVM_REGISTER_GLOBAL("relax.CopyWithNewParams").set_body_typed(CopyWithNewParams);

/*! \brief Helper to implement extend function.*/
class ExtendFuncMutator : public ExprMutator {
 public:
  explicit ExtendFuncMutator(const SeqExpr& ex_body) : ex_body_(ex_body) {}

  Expr VisitExpr_(const SeqExprNode* seq_expr) override {
    // mutate only the last block.
    Array<BindingBlock> blocks;
    for (int i = 0; i < static_cast<int>(seq_expr->blocks.size()); ++i) {
      if (i < static_cast<int>(seq_expr->blocks.size()) - 1) {
        blocks.push_back(seq_expr->blocks[i]);
      } else {
        BindingBlock new_block = this->VisitBindingBlock(seq_expr->blocks[i]);
        if (!new_block->bindings.empty()) {
          blocks.push_back(new_block);
        }
      }
    }
    this->VisitExpr(seq_expr->body);
    return SeqExpr(blocks, ex_body_->body);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) override {
    builder_->BeginDataflowBlock();
    // emit original bindings.
    for (const auto& binding : block->bindings) {
      this->VisitBinding(binding);
    }

    ICHECK(orig_rets_var_.size() == orig_rets.size());
    for (int i = 0; i < static_cast<int>(orig_rets_var_.size()); ++i) {
      if (orig_rets_var_[i].defined()) {
        builder_->EmitNormalized(VarBinding(orig_rets_var_[i].value(), orig_rets[i]));
      }
    }

    // emit blocks for extend part.
    for (BindingBlock block : ex_body_->blocks) {
      for (Binding binding : block->bindings) {
        this->VisitBinding(binding);
      }
    }

    return builder_->EndBlock();
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    Var new_var = Downcast<Var>(this->VisitExpr(binding->var));
    Expr new_value = this->VisitExpr(binding->value);
    builder_->EmitNormalized(VarBinding(new_var, new_value));
  }

  // remap orignal dataflow var
  // TODO(chaofan): a better way to check whether new_ret_var should be dataflow
  void RemapToDataflow(SeqExpr body) {
    for (BindingBlock block : body->blocks) {
      for (Binding binding : block->bindings) {
        const auto* binding_node = binding.as<VarBindingNode>();
        if (binding_node && !binding_node->var->IsInstance<DataflowVarNode>()) {
          Var new_binding_var = DataflowVar(
              binding_node->var->vid, GetStructInfo(binding_node->var), binding_node->var->span);
          this->var_remap_[binding_node->var->vid] = new_binding_var;
        }
      }
    }
  }

  Array<Var> RemapExParams(const Array<Var>& ex_func_params, Array<Var> new_params) {
    for (int i = 0; i < static_cast<int>(ex_func_params.size()); ++i) {
      Var ex_param = ex_func_params[i];
      if (i < static_cast<int>(orig_rets.size())) {
        // map return value to ex param
        if (const auto* var_node = orig_rets[i].as<VarNode>()) {
          ICHECK(orig_rets[i].as<DataflowVarNode>());
          orig_rets_var_.push_back(NullOpt);
          this->var_remap_[ex_param->vid] = GetRef<Var>(var_node);
        } else {
          Var new_ret_var =
              DataflowVar(/*name_hint=*/"ret_" + std::to_string(i), GetStructInfo(orig_rets[i]));
          orig_rets_var_.push_back(new_ret_var);
          this->var_remap_[ex_param->vid] = new_ret_var;
        }
      } else {
        // append to the param list
        Var new_ex_param = Var(ex_param->vid, GetStructInfo(ex_param), ex_param->span);
        this->var_remap_[ex_param->vid] = new_ex_param;
        new_params.push_back(new_ex_param);
      }
    }
    return new_params;
  }

  Array<Expr> orig_rets;

 private:
  SeqExpr ex_body_;
  Array<Optional<Var>> orig_rets_var_;
};

/*!
 * \brief Extend a relax function by another given function.
 * \param orig_func The function to be extended.
 * \param ex_func The function to be linked after the orig_func.
 * \return The result function after extending.
 */
Function ExtendFunc(Function orig_func, Function ex_func) {
  CHECK(orig_func->body->IsInstance<SeqExprNode>())
      << "the body of the original function is not SeqExpr.";
  CHECK(ex_func->body->IsInstance<SeqExprNode>()) << "the body of the ex function is not SeqExpr.";

  auto param_copied_func = CopyWithNewParams(orig_func);
  auto seq_expr = Downcast<SeqExpr>(param_copied_func->body);

  ExtendFuncMutator mutator(Downcast<SeqExpr>(ex_func->body));
  mutator.RemapToDataflow(seq_expr);
  // Get the orignal rets. If it is a Tuple, unpack it.
  if (orig_func->ret_struct_info.as<TupleStructInfoNode>()) {
    const auto* tuple_node = seq_expr->body.as<TupleNode>();
    ICHECK(tuple_node != nullptr);
    for (Expr field : tuple_node->fields) {
      mutator.orig_rets.push_back(mutator.VisitExpr(field));
    }
  } else {
    mutator.orig_rets.push_back(mutator.VisitExpr(seq_expr->body));
  }

  CHECK(ex_func->params.size() >= mutator.orig_rets.size())
      << "The number of return values of original functions should be greater than the number of "
         "parameters of ex function";

  auto new_params = mutator.RemapExParams(ex_func->params, param_copied_func->params);
  Expr new_body = mutator.VisitExpr(seq_expr);
  return Function(new_params, new_body, ex_func->ret_struct_info, param_copied_func->attrs);
}

TVM_REGISTER_GLOBAL("relax.ExtendFunc").set_body_typed(ExtendFunc);

}  // namespace relax
}  // namespace tvm
