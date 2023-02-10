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
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>
      bindings_lifted;  // the bindings of the original function that are lifted
  std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>
      output_to_index_;  // the index of the original bindings in the output tuple
};

class TransformParamsFuncBuilder : public ExprMutator {
 public:
  TransformParamsFuncBuilder() { builder_->BeginDataflowBlock(); }

  void AddInput(const Var& var) { inputs_.push_back(var); }

  void AddBinding(const VarBinding& binding) { bindings_.push_back(binding); }

  void MarkOutput(const Var& output) { outputs_.insert(output); }

  std::pair<Function, std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>> Build() {
    Array<StructInfo> input_sinfo;
    Array<Expr> output_vars;
    std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual> output_to_index;

    for (const auto& input : inputs_) {
      input_sinfo.push_back(Downcast<StructInfo>(input->struct_info_.value()));
    }
    Var params("params", TupleStructInfo(input_sinfo));

    for (const auto& input : inputs_) {
      input_remap_.emplace(input.get(), TupleGetItem(params, input_remap_.size()));
    }

    for (const auto& binding : bindings_) {
      if (outputs_.count(binding->var)) {
        auto output_var = builder_->Emit(VisitExpr(binding->value));
        var_remap_[binding->var->vid] = output_var;
        output_to_index[binding->var] = output_vars.size();
        output_vars.push_back(output_var);
      } else {
        // builder_->Emit(Visit(binding));
        VisitBinding(binding);
      }
    }

    Expr transformed_params = builder_->EmitOutput(Tuple(output_vars));
    BindingBlock block = builder_->EndBlock();
    Expr body = builder_->Normalize(SeqExpr({block}, transformed_params));
    Function f_transform_params =
        Function(/*params=*/{params}, /*body=*/body, /*ret_struct_info=*/NullOpt);
    LOG(INFO) << f_transform_params;
    return {f_transform_params, output_to_index};
  }

  Expr VisitExpr_(const VarNode* var) final {
    if (auto it = input_remap_.find(var); it != input_remap_.end()) {
      return builder_->Emit((*it).second);
    } else {
      return ExprMutator::VisitExpr_(var);
    }
  }

  Array<Var> inputs_;
  std::unordered_map<const VarNode*, Expr> input_remap_;
  Array<VarBinding> bindings_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> outputs_;
};

class LiftTransformParamsPlanner : public ExprVisitor {
 public:
  LiftTransformParamsInfo Plan(const Function& function, int num_inputs) {
    for (int i = num_inputs; i < static_cast<int>(function->params.size()); ++i) {
      builder_.AddInput(function->params[i]);
      lifted_bindings_.emplace(function->params[i]);
    }
    VisitExpr(function->body);

    const auto& [f_transform_params, output_to_index] = builder_.Build();
    return {f_transform_params, std::move(lifted_bindings_), output_to_index};
  }

 private:
  void VisitBinding_(const VarBindingNode* binding) final {
    std::vector<const VarNode*> producers;
    bool can_lift = true;
    PostOrderVisit(binding->value, [&](const ObjectRef& obj) {
      if (const VarNode* var = obj.as<VarNode>()) {
        producers.push_back(var);
        if (!lifted_bindings_.count(GetRef<Var>(var))) {
          LOG(INFO) << "cannot lift " << obj;
          can_lift = false;
        }
      }
    });
    if (can_lift) {
      lifted_bindings_.insert(binding->var);
      builder_.AddBinding(GetRef<VarBinding>(binding));
    } else {
      for (const VarNode* producer : producers) {
        if (lifted_bindings_.count(GetRef<Var>(producer))) {
          builder_.MarkOutput(GetRef<Var>(producer));
        }
      }
    }
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> lifted_bindings_;

  TransformParamsFuncBuilder builder_;
};

class TransformParamsLifter : public ExprMutator {
 public:
  explicit TransformParamsLifter(const IRModule& module) : ExprMutator(module) {}

  IRModule Lift() {
    auto mod = builder_->GetContextIRModule();
    GlobalVar gv_main = mod->GetGlobalVar("main");
    Function func = Downcast<Function>(mod->Lookup(gv_main));
    func = RewriteFunc(func);
    builder_->UpdateFunction(gv_main, func);
    return builder_->GetContextIRModule();
  }

  Function RewriteFunc(const Function& func) {
    auto opt_num_input = func->attrs.GetAttr<Integer>("num_input");
    if (!opt_num_input.defined()) {
      return func;
    }
    LiftTransformParamsPlanner planner;
    int64_t params_begin = opt_num_input.value()->value;
    lift_info_ = planner.Plan(func, params_begin);

    // Step 1: Add f_transform_params to the module

    // TransformParamsFinder finder;
    // int params_end = func->params.size();
    // lift_info_ = finder.Find(func, params_begin, params_end);
    builder_->AddFunction(lift_info_.f_transform_params, "transform_params");

    // Update the current function with the transformed parameters

    // Step 1: Update the function signature
    Var params("params", lift_info_.f_transform_params->ret_struct_info);
    Array<Var> new_params;
    for (int i = 0; i < params_begin; ++i) {
      new_params.push_back(func->params[i]);
    }
    new_params.push_back(params);

    for (const auto& [var, index] : lift_info_.output_to_index_) {
      param_remap_[var] = TupleGetItem(params, index);
    }

    // Step 2: Update the function body
    auto new_body = VisitExpr(func->body);

    auto new_attrs = func->attrs;
    auto* new_attrs_node = new_attrs.CopyOnWrite();
    new_attrs_node->dict.erase("num_input");
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
