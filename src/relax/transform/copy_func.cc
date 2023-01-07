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
 * \file src/relax/transform/copy_func.cc
 * \brief Copy the specified function in the given IRModule, with the given name.
 */
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class FuncCopier : public ExprMutator {
 public:
  explicit FuncCopier(const IRModule& mod, const GlobalVar& gvar, const String& new_name)
      : ExprMutator(mod), mod_(mod), gvar_(gvar), new_name_(std::move(new_name)) {}

  IRModule Transform() {
    auto new_module = GetRef<IRModule>(mod_.CopyOnWrite());

    auto func_before = Downcast<Function>(mod_->Lookup(gvar_));
    auto func_after_var = GlobalVar(new_name_);
    auto func_after = Downcast<Function>(this->VisitExpr(func_before));
    new_module->Add(func_after_var, func_after);
    return new_module;
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    // the parameters would be copied and substituted to satisfy the restriction in the well-formed
    // check: any two functions cannot share the same parameter variable.
    Array<Var> new_params;
    for (Var param : func->params) {
      Var new_param = Var(param->vid, GetStructInfo(param), param->span);
      this->var_remap_[param->vid] = new_param;
      new_params.push_back(new_param);
    }

    Expr body = this->VisitWithNewScope(func->body, new_params);

    return Function(new_params, body, func->ret_struct_info, func->attrs);
  }

  // inputs
  IRModule mod_;
  GlobalVar gvar_;
  String new_name_;
};

/*!
 * \brief The internal function of tvm::relax::transform::CopyFunc.
 * \param mod The module.
 * \param gvar The GlobalVar of the specified function.
 * \param new_name The name of the copied function.
 * \return The module after transformation.
 */
IRModule CopyFunc(const IRModule& mod, const GlobalVar& gvar, Optional<String> new_name) {
  auto* func = mod->Lookup(gvar).as<FunctionNode>();
  CHECK(func) << "relax function " << gvar->name_hint << " is not found";

  if (!new_name.defined()) {
    new_name = gvar->name_hint + "_new";
  }

  return FuncCopier(mod, gvar, new_name.value()).Transform();
}

namespace transform {

Pass CopyFunc(GlobalVar global_var, Optional<String> new_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return relax::CopyFunc(mod, global_var, new_name); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"CopyFunc",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.CopyFunc").set_body_typed(CopyFunc);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
