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
 * \file tvm/relax/transform/lambda_lift.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>

#include <iostream>
#include <vector>

namespace tvm {
namespace relax {

struct LiftTransformParamsInfo {
  Function f_transform_params;  // the lifted function that transforms the parameters
  std::unordered_set<Var, ObjectHash, ObjectEqual>
      bindings_lifted;  // the bindings of the original function that are lifted
  std::unordered_map<Var, int, ObjectHash, ObjectEqual>
      output_to_index_;  // the index of the original bindings in the output tuple
};

Expr Substitute(const Expr& expr, const Map<Var, Var>& subst_map) {
  struct Substitutor : ExprMutator {
    Map<Var, Var> subst_map_;

    explicit Substitutor(const Map<Var, Var>& subst_map) : subst_map_(subst_map) {}

    Expr VisitExpr_(const VarNode* var) final {
      if (auto it = subst_map_.find(GetRef<Var>(var)); it != subst_map_.end()) {
        return (*it).second;
      } else {
        return GetRef<Expr>(var);
      }
    }
  };
  return Substitutor(subst_map).VisitExpr(std::move(expr));
}

class TransformParamsFinder : public ExprVisitor {
 public:
  LiftTransformParamsInfo Find(const Function& func, int begin, int end) {
    Array<StructInfo> param_sinfo;
    for (int i = begin; i < end; ++i) {
      param_sinfo.push_back(Downcast<StructInfo>(func->params[i]->struct_info_.value()));
    }

    builder_ = BlockBuilder::Create(NullOpt);
    Var param_tuple("params", TupleStructInfo(param_sinfo));
    builder_->BeginDataflowBlock();
    for (size_t i = 0; i < param_sinfo.size(); ++i) {
      auto new_var = builder_->Emit(TupleGetItem(param_tuple, i));
      var_remap_.Set(func->params[begin + i], new_var);
      bindings_.insert(func->params[begin + i]);
    }

    VisitExpr(func);
    auto output = builder_->EmitOutput(Tuple(outputs));
    BindingBlock block = builder_->EndBlock();
    Expr seq = builder_->Normalize(SeqExpr({block}, output));

    Function f_transform_params{
        {param_tuple}, seq, Downcast<StructInfo>(output->struct_info_.value())};
    return {f_transform_params, bindings_, output_to_index_};
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    if (val->op.same_as(call_tir_op_)) {
      if (std::all_of(val->args.begin(), val->args.end(), [&](const Expr& arg) {
            const auto* var = arg.as<VarNode>();
            return !var || bindings_.count(GetRef<Var>(var));
          })) {
        bindings_.insert(binding->var);
        output_to_index_[binding->var] = outputs.size();

        auto new_var = builder_->Emit(Substitute(binding->value, var_remap_));
        // TODO(wuwei): we can prune the outputs and only keep the final result of transformations
        // when there are a sequence of transformations applied to the parameters.
        outputs.push_back(new_var);
      }
    }
  }

 private:
  std::unordered_set<Var, ObjectHash, ObjectEqual> bindings_;
  BlockBuilder builder_;

  Array<Expr> outputs;
  Map<Var, Var> var_remap_;
  std::unordered_map<Var, int, ObjectHash, ObjectEqual> output_to_index_;
};

class TransformParamsLifter : public ExprMutator {
 public:
  explicit TransformParamsLifter(const IRModule& module) : ExprMutator(module) {}

  IRModule Lift() {
    auto mod = builder_->GetContextIRModule();
    GlobalVar gv_main =
        mod->GetGlobalVar("main");  // TODO(wuwei): handle the case entry function is not main
    Function func = Downcast<Function>(mod->Lookup(gv_main));
    func = RewriteFunc(func);
    builder_->UpdateFunction(gv_main, func);
    return builder_->GetContextIRModule();
  }

  Function RewriteFunc(const Function& func) {
    auto opt_param_begin = func->attrs.GetAttr<Integer>("param_begin");
    auto opt_param_end = func->attrs.GetAttr<Integer>("param_end");
    if (!opt_param_begin.defined() || !opt_param_end.defined()) {
      return func;
    }
    int64_t params_begin = opt_param_begin.value()->value;
    int64_t params_end = opt_param_end.value()->value;

    TransformParamsFinder finder;
    lift_info_ = finder.Find(func, params_begin, params_end);
    builder_->AddFunction(lift_info_.f_transform_params, "transform_params");

    // Update the current function with the transformed parameters

    // Step 1: Update the function signature
    Var params("params", lift_info_.f_transform_params->ret_struct_info);
    Array<Var> new_params;
    for (int i = 0; i < params_begin; ++i) {
      new_params.push_back(func->params[i]);
    }
    new_params.push_back(params);
    for (int i = params_end; i < static_cast<int>(func->params.size()); ++i) {
      new_params.push_back(func->params[i]);
    }

    for (const auto& [var, index] : lift_info_.output_to_index_) {
      param_remap_[var] = TupleGetItem(params, index);
    }

    // Step 2: Update the function body
    auto new_body = VisitExpr(func->body);

    auto new_attrs = func->attrs;
    auto* new_attrs_node = new_attrs.CopyOnWrite();
    new_attrs_node->dict.erase("param_begin");
    new_attrs_node->dict.erase("param_end");
    if (new_attrs->dict.empty()) {
      new_attrs = NullValue<DictAttrs>();
    }

    Function new_func(new_params, new_body, func->ret_struct_info, new_attrs);
    return new_func;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) final {
    if (lift_info_.bindings_lifted.count(binding->var)) {
      return;
    }
    ExprMutator::VisitBinding_(binding, val);
  }

  Expr VisitExpr_(const VarNode* var) final {
    auto it = param_remap_.find(GetRef<Var>(var));
    if (it != param_remap_.end()) {
      return builder_->Emit(it->second);
    }
    return ExprMutator::VisitExpr_(var);
  }

  Expr VisitExpr_(const DataflowVarNode* var) final {
    return VisitExpr_(static_cast<const VarNode*>(var));
  }

 private:
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> param_remap_;
  LiftTransformParamsInfo lift_info_;
};

namespace transform {
Pass LiftTransformParams() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return TransformParamsLifter(m).Lift(); };
  return CreateModulePass(pass_func, 1, "LiftTransformParams", {});
}

TVM_REGISTER_GLOBAL("relax.transform.LiftTransformParams").set_body_typed(LiftTransformParams);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
