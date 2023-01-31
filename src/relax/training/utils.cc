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

#include "utils.h"

namespace tvm {
namespace relax {

/*! \brief Helper to implement append loss.*/
class AppendLossMutator : public ExprMutator {
 public:
  explicit AppendLossMutator(const SeqExpr& loss_body) : loss_body_(loss_body) {}

  Expr VisitExpr_(const SeqExprNode* seq_expr) final {
    // mutate only the last block.
    Array<BindingBlock> blocks;
    for (int i = 0; i < static_cast<int>(seq_expr->blocks.size()); ++i) {
      CHECK(seq_expr->blocks[i].as<DataflowBlockNode>())
          << "All blocks in original functions should be Dataflow Block.";
      if (i < static_cast<int>(seq_expr->blocks.size()) - 1) {
        blocks.push_back(seq_expr->blocks[i]);
      } else {
        BindingBlock new_block = this->VisitBindingBlock(seq_expr->blocks[i]);
        if (!new_block->bindings.empty()) {
          blocks.push_back(new_block);
        }
      }
    }
    return SeqExpr(blocks, loss_body_->body);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
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

    // emit blocks for loss function part.
    for (const BindingBlock& block : loss_body_->blocks) {
      CHECK(block.as<DataflowBlockNode>())
          << "All blocks in loss functions should be Dataflow Block.";
      for (const Binding& binding : block->bindings) {
        this->VisitBinding(binding);
      }
    }

    return builder_->EndBlock();
  }

  // Remap the def site var using VisitExpr.
  Var VisitVarDef(const Var& var) final { return Downcast<Var>(this->VisitExpr(var)); }

  // Remap original dataflow var.
  // TODO(chaofan): a better way to check whether new_ret_var should be dataflow
  void RemapToDataflow(SeqExpr body) {
    const auto& block = body->blocks.back();
    for (const Binding& binding : block->bindings) {
      const auto* binding_node = binding.as<VarBindingNode>();
      if (binding_node && !binding_node->var->IsInstance<DataflowVarNode>()) {
        Var new_binding_var = DataflowVar(binding_node->var->vid, GetStructInfo(binding_node->var),
                                          binding_node->var->span);
        this->var_remap_[binding_node->var->vid] = new_binding_var;
      }
    }
  }

  Array<Var> RemapLossParams(const Array<Var>& loss_func_params, Array<Var> new_params) {
    for (int i = 0; i < static_cast<int>(loss_func_params.size()); ++i) {
      Var loss_param = loss_func_params[i];
      auto loss_param_sinfo = GetStructInfo(loss_param);

      if (i < static_cast<int>(orig_rets.size())) {  // map return value to loss param
        auto orig_ret_sinfo = GetStructInfo(orig_rets[i]);
        ICHECK(structural_equal_(orig_ret_sinfo, loss_param_sinfo))
            << "The struct info of the " << i
            << "-th return value of orig func is: " << orig_ret_sinfo
            << " while the corresponding struct info of parameter of loss function is "
            << loss_param_sinfo << ", which is different.";

        if (const auto* var_node = orig_rets[i].as<VarNode>()) {
          ICHECK(orig_rets[i].as<DataflowVarNode>());
          orig_rets_var_.push_back(NullOpt);
          this->var_remap_[loss_param->vid] = GetRef<Var>(var_node);
        } else {
          Var new_ret_var = DataflowVar(/*name_hint=*/"ret_" + std::to_string(i), orig_ret_sinfo);
          orig_rets_var_.push_back(new_ret_var);
          this->var_remap_[loss_param->vid] = new_ret_var;
        }
      } else {  // append to the param list
        Var new_loss_param = Var(loss_param->vid, loss_param_sinfo, loss_param->span);
        this->var_remap_[loss_param->vid] = new_loss_param;
        new_params.push_back(new_loss_param);
      }
    }
    return new_params;
  }

  /*! \brief The original unpacked rets. */
  Array<Expr> orig_rets;

 private:
  /*! \brief The body of the loss function */
  SeqExpr loss_body_;
  /*! \brief The var created for original rets. NullOpt if the original ret is already a var. */
  Array<Optional<Var>> orig_rets_var_;
  /*! \brief The structural equality checker */
  StructuralEqual structural_equal_;
};

/*!
 * \brief Local helper to append a specified loss function after the original function.
 * \param orig_func The function to be appended to.
 * \param loss_func The loss function.
 * \return The result function after appended.
 */
Function AppendLoss(Function orig_func, Function loss_func) {
  CHECK(orig_func->body->IsInstance<SeqExprNode>())
      << "The body of the original function is expected to be a SeqExpr, but got"
      << orig_func->body->GetTypeKey();
  CHECK(loss_func->body->IsInstance<SeqExprNode>())
      << "The body of the loss function is expected to be a SeqExpr, but got"
      << loss_func->body->GetTypeKey();

  Function param_copied_func = CopyWithNewParams(orig_func);
  auto seq_expr = Downcast<SeqExpr>(param_copied_func->body);

  AppendLossMutator mutator(Downcast<SeqExpr>(loss_func->body));
  mutator.RemapToDataflow(seq_expr);
  // Get the original rets. If it is a Tuple, unpack it.
  if (orig_func->ret_struct_info.as<TupleStructInfoNode>()) {
    const auto* tuple_node = seq_expr->body.as<TupleNode>();
    ICHECK(tuple_node != nullptr);
    for (const Expr& field : tuple_node->fields) {
      mutator.orig_rets.push_back(mutator.VisitExpr(field));
    }
  } else {
    mutator.orig_rets.push_back(mutator.VisitExpr(seq_expr->body));
  }

  CHECK(loss_func->params.size() >= mutator.orig_rets.size())
      << "The number of return values of original functions should be greater than the number of "
         "parameters of loss function. Got "
      << mutator.orig_rets.size() << " > " << loss_func->params.size();

  Array<Var> new_params = mutator.RemapLossParams(loss_func->params, param_copied_func->params);
  Expr new_body = mutator.VisitExpr(seq_expr);
  return Function(std::move(new_params), std::move(new_body), loss_func->ret_struct_info,
                  param_copied_func->attrs);
}

TVM_REGISTER_GLOBAL("relax.training.AppendLoss").set_body_typed(AppendLoss);

}  // namespace relax
}  // namespace tvm
