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

/*! \file src/relax/transform/simplify_norm.cc */

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {
namespace {
TensorStructInfo MatchTensorStructInfo(Expr data) {
  auto _sinfo = MatchStructInfo<TensorStructInfo>(data);
  ICHECK(_sinfo.defined()) << "Expect data to be a tensor, but get " << GetStructInfo(data);
  return _sinfo.value();
}

Expr ExpandToMatchInput(Expr data, int ndim, Array<Integer> axes) {
  axes = GetOrderedPositiveAxes(axes, ndim);
  Array<Integer> expand_axes;
  for (int i = 0, j = 0; i < ndim; ++i) {
    if (j < static_cast<int>(axes.size()) && i == axes[j]->value) {
      ++j;
    } else {
      expand_axes.push_back(i);
    }
  }
  return expand_dims(data, expand_axes);
}

Tuple SimplifyBatchNormInference(const CallNode* call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  ICHECK_NOTNULL(attrs);

  Expr data = call->args[0];
  TensorStructInfo sinfo = MatchTensorStructInfo(data);
  Expr gamma = call->args[1];
  Expr beta = call->args[2];

  Expr moving_mean = ExpandToMatchInput(call->args[3], sinfo->ndim, {attrs->axis});
  Expr moving_var = ExpandToMatchInput(call->args[4], sinfo->ndim, {attrs->axis});

  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(attrs->epsilon, sinfo->dtype);
  Expr sqrt_var = sqrt(add(moving_var, epsilon));
  Expr out = divide(subtract(data, moving_mean), sqrt_var);

  if (attrs->scale) {
    out = multiply(out, ExpandToMatchInput(gamma, sinfo->ndim, {attrs->axis}));
  }
  if (attrs->center) {
    out = add(out, ExpandToMatchInput(beta, sinfo->ndim, {attrs->axis}));
  }

  return Tuple({out, call->args[3], call->args[4]});
}

Tuple SimplifyBatchNormTraining(const CallNode* call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  ICHECK_NOTNULL(attrs);

  Expr data = call->args[0];
  TensorStructInfo sinfo = MatchTensorStructInfo(data);
  Expr gamma = call->args[1];
  Expr beta = call->args[2];

  Array<Integer> reduce_axes;
  for (int i = 0; i < sinfo->ndim; ++i) {
    if (i != attrs->axis) {
      reduce_axes.push_back(i);
    }
  }

  Expr data_mean = mean(data, reduce_axes, false);
  Expr data_mean_rs = ExpandToMatchInput(data_mean, sinfo->ndim, {attrs->axis});
  Expr data_var = variance(data, reduce_axes, false);
  Expr data_var_rs = ExpandToMatchInput(data_var, sinfo->ndim, {attrs->axis});

  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(attrs->epsilon, sinfo->dtype);
  Expr sqrt_var = sqrt(add(data_var_rs, epsilon));
  Expr out = divide(subtract(data, data_mean_rs), sqrt_var);

  if (attrs->scale) {
    out = multiply(out, ExpandToMatchInput(gamma, sinfo->ndim, {attrs->axis}));
  }
  if (attrs->center) {
    out = add(out, ExpandToMatchInput(beta, sinfo->ndim, {attrs->axis}));
  }

  Expr moving_mean = call->args[3];
  Expr moving_var = call->args[4];
  Expr momentum = MakeConstantScalar(attrs->momentum, sinfo->dtype);
  Expr one_minus_mom = MakeConstantScalar(1 - attrs->momentum, sinfo->dtype);

  return Tuple({
      out,
      add(multiply(one_minus_mom, moving_mean), multiply(momentum, data_mean)),
      add(multiply(one_minus_mom, moving_var), multiply(momentum, data_var)),
  });
}

/*! \brief A mutator to simplify the normalization. */
class NormSimplifier : private ExprMutator {
 public:
  constexpr static const char* kModeEval = "eval";
  constexpr static const char* kModeTraining = "training";

  static IRModule Simplify(IRModule mod, Optional<String> func_name, String mode) {
    CHECK(mode == kModeEval || mode == kModeTraining)
        << "The argument mode must be one of the following values: \"eval\", \"training\".";

    auto simplifier = NormSimplifier(mode);

    IRModuleNode* new_module = mod.CopyOnWrite();

    if (!func_name.defined()) {  // simplify all functions
      Map<GlobalVar, BaseFunc> functions = mod->functions;
      for (const auto& func_pr : functions) {
        if (const auto* relax_f = func_pr.second.as<FunctionNode>()) {
          Function f_simplified = Downcast<Function>(simplifier(GetRef<Function>(relax_f)));
          new_module->Update(func_pr.first, f_simplified);
        }
      }
    } else {  // simplify specified function
      auto* func_ptr = mod->Lookup(func_name.value()).as<FunctionNode>();
      CHECK(func_ptr) << func_name.value() << "is not a Relax Function";
      auto gvar = mod->GetGlobalVar(func_name.value());
      auto func = GetRef<Function>(func_ptr);
      new_module->Update(gvar, Downcast<Function>(simplifier(func)));
    }

    return GetRef<IRModule>(new_module);
  }

 private:
  explicit NormSimplifier(String mode) : ExprMutator(), mode_(mode) {}

  Expr VisitExpr_(const CallNode* call) final {
    if (call->op == Op::Get("relax.nn.batch_norm")) {
      if (mode_ == kModeEval) {
        return SimplifyBatchNormInference(call);
      } else {
        ICHECK_EQ(mode_, kModeTraining);
        return SimplifyBatchNormTraining(call);
      }
    }
    return GetRef<Call>(call);
  }

  const String mode_;
};
}  // namespace

namespace transform {
Pass SimplifyNorm(Optional<String> func_name, String mode) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return NormSimplifier::Simplify(mod, func_name, mode); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"SimplifyNorm",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.SimplifyNorm").set_body_typed(SimplifyNorm);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
