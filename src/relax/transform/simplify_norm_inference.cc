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

/*! \file src/relax/transform/simplify_norm_inference.cc */

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

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

/*!
 * \brief The common part for unpacking normalization
 * \param dev The Expr of data - mean (may avoid duplicate computation)
 * \param var The Expr of variance
 * \param gamma The Expr of gamma, skip scaling if not defined.
 * \param beta The Expr of beta, skip shifting if not defined.
 * \param axes The axes to reduce.
 * \param eps The epsilon value.
 * \param ndim The ndim of the input data.
 * \param dtype The dtype of the input data.
 * \return The Expr of the unpacked normalization.
 */
Expr SimplifyCommonNorm(Expr dev, Expr var, Optional<Expr> gamma, Optional<Expr> beta,
                        Array<Integer> axes, float eps, int ndim, DataType dtype) {
  // output = (x - mean) / sqrt(var + epsilon) * gamma + beta
  Expr epsilon = MakeConstantScalar(eps, dtype);
  Expr sqrt_var = sqrt(add(var, epsilon));
  Expr out = divide(dev, sqrt_var);

  if (gamma.defined()) {
    out = multiply(out, ExpandToMatchInput(gamma.value(), ndim, axes));
  }
  if (beta.defined()) {
    out = add(out, ExpandToMatchInput(beta.value(), ndim, axes));
  }
  return out;
}

std::pair<Expr, Expr> ComputeDevVar(Expr data, Array<Integer> axes) {
  Expr mean_expr = mean(data, axes, /*keepdims=*/true);
  Expr dev = subtract(data, mean_expr);
  Expr var = mean(multiply(dev, dev), axes, /*keepdims=*/true);
  return std::make_pair(dev, var);
}

Expr SimplifyBatchNorm(const CallNode* call) {
  auto attrs = call->attrs.as<BatchNormAttrs>();
  ICHECK_NOTNULL(attrs);
  Optional<Expr> gamma = attrs->scale ? call->args[1] : Optional<Expr>(NullOpt);
  Optional<Expr> beta = attrs->center ? call->args[2] : Optional<Expr>(NullOpt);
  TensorStructInfo sinfo = MatchTensorStructInfo(call->args[0]);
  Expr moving_mean = ExpandToMatchInput(call->args[3], sinfo->ndim, {attrs->axis});
  Expr moving_var = ExpandToMatchInput(call->args[4], sinfo->ndim, {attrs->axis});
  Expr dev = subtract(call->args[0], moving_mean);
  return SimplifyCommonNorm(dev, moving_var, gamma, beta, {attrs->axis}, attrs->epsilon,
                            sinfo->ndim, sinfo->dtype);
}

Expr SimplifyLayerNorm(const CallNode* call) {
  auto attrs = call->attrs.as<LayerNormAttrs>();
  ICHECK_NOTNULL(attrs);
  Optional<Expr> gamma = attrs->scale ? call->args[1] : Optional<Expr>(NullOpt);
  Optional<Expr> beta = attrs->center ? call->args[2] : Optional<Expr>(NullOpt);
  auto res = ComputeDevVar(call->args[0], attrs->axes);
  const Expr& dev = res.first;
  const Expr& var = res.second;
  TensorStructInfo sinfo = MatchTensorStructInfo(call->args[0]);
  return SimplifyCommonNorm(dev, var, gamma, beta, attrs->axes, attrs->epsilon, sinfo->ndim,
                            sinfo->dtype);
}

/*! \brief A mutator to simplify the normalization inference. */
class NormInferenceSimplifier : public ExprMutator {
 public:
  static Expr Simplify(Expr expr) { return NormInferenceSimplifier()(expr); }

 private:
  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr expr = ExprMutator::VisitExpr_(op);
    op = expr.as<TupleGetItemNode>();
    ICHECK_NOTNULL(op);

    auto it = batch_norm_map_.find(op->tuple);
    if (it != batch_norm_map_.end() && op->index == 0) {
      return (*it).second;
    } else {
      return expr;
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    Expr expr = ExprMutator::VisitExpr_(call);
    call = expr.as<CallNode>();
    ICHECK_NOTNULL(call);

    static const Op& batch_norm_op = Op::Get("relax.nn.batch_norm");
    static const Op& layer_norm_op = Op::Get("relax.nn.layer_norm");
    if (call->op == batch_norm_op) {
      // NOTE: we won't directly replace the batch_norm call since
      // the following bindings may depend on the returned moving_mean and moving_var.
      // Instead, we will store the unpacked value in the batch_norm_map_, and replace it
      // at the TupleGetItemNode. And the original batch_norm call will be removed in the
      // follow-up pass `RemoveAllUnused`
      return expr;
    } else if (call->op == layer_norm_op) {
      return SimplifyLayerNorm(call);
    } else {
      return expr;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* val) final {
    ExprMutator::VisitBinding_(binding, val);
    if (val->op == Op::Get("relax.nn.batch_norm")) {
      batch_norm_map_.Set(binding->var, SimplifyBatchNorm(val));
    }
  }

 private:
  /*! \brief The mapping from binding var of batch_norm to the unpacked value. */
  Map<Expr, Expr> batch_norm_map_;
};

namespace transform {
Pass SimplifyNormInference() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        f = Downcast<Function>(NormInferenceSimplifier::Simplify(f));
        // Remove original batch_norm op if it's not used.
        return RemoveAllUnused(f);
      };
  return CreateFunctionPass(/*pass_function=*/pass_func,            //
                            /*opt_level=*/0,                        //
                            /*pass_name=*/"SimplifyNormInference",  //
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.SimplifyNormInference").set_body_typed(SimplifyNormInference);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
