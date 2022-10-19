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
 * \file reduce.cc
 * \brief Reduction operators.
 */

#include "reduce.h"

#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(ReduceAttrs);

Optional<Expr> InferShapeReduction(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Reduction op should have 1 argument1");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  if (shape == nullptr) {
    return RuntimeDepShape();
  }
  int ndim = shape->values.size();

  const auto* attrs = call->attrs.as<ReduceAttrs>();
  Array<Integer> axis;
  if (attrs->axis.defined()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(ndim);
    for (int i = 0; i < ndim; ++i) {
      axis.push_back(Integer(i));
    }
  }
  int n_axis = axis.size();

  std::vector<bool> appeared_axes;
  appeared_axes.resize(ndim);
  for (int i = 0; i < n_axis; ++i) {
    int dim = axis[i]->value;
    if (dim < 0) {
      dim = ndim + dim;
    }
    if (dim < 0 || dim >= ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Reduction operator expects all input axis indices to be in range ["
                         << -ndim << ", " << ndim << "). However, the axis index " << i << " is "
                         << axis[i]->value << ", which is out of range");
    }
    if (appeared_axes[dim]) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Reduction operator expects all input axis indices not to duplicate. "
                            "However, there are at least two indices correspond to dim "
                         << dim);
    }
    appeared_axes[dim] = true;
  }

  Array<PrimExpr> output_shape;
  output_shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    if (!appeared_axes[i]) {
      output_shape.push_back(shape->values[i]);
    } else if (attrs->keepdims) {
      output_shape.push_back(tir::make_const(DataType::Int(32), 1));
    }
  }
  return ShapeExpr(std::move(output_shape));
}

Type InferTypeReduction(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Reduction op should have 1 argument1");
  }

  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Reduction operator expects the input to have DynTensorType. However, "
                          "the given input has type "
                       << call->args[0]->checked_type()->GetTypeKey());
  }
  const auto* attrs = call->attrs.as<ReduceAttrs>();

  if (type->IsUnknownNdim()) {
    if (attrs->axis.defined()) {
      int n_axis = attrs->axis.value().size();
      std::unordered_set<int> non_negative_indices;
      std::unordered_set<int> negative_indices;
      non_negative_indices.reserve(n_axis);
      negative_indices.reserve(n_axis);
      for (int i = 0; i < n_axis; ++i) {
        int idx = attrs->axis.value()[i]->value;
        std::unordered_set<int>* set = idx >= 0 ? &non_negative_indices : &negative_indices;
        std::pair<std::unordered_set<int>::iterator, bool> insert_result = set->insert(idx);
        if (!insert_result.second) {
          diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                             << "Reduction operator expects all input axis indices not to "
                                "duplicate. However, there are at least two indices of "
                             << idx);
        }
      }
      return DynTensorType(-1, type->dtype);
    } else {
      return DynTensorType(attrs->keepdims ? -1 : 0, type->dtype);
    }
  }

  int ndim = type->ndim;
  if (!attrs->axis.defined()) {
    return DynTensorType(attrs->keepdims ? ndim : 0, type->dtype);
  }

  int n_axis = attrs->axis.value().size();
  std::vector<bool> appeared_axes;
  appeared_axes.resize(ndim);
  for (int i = 0; i < n_axis; ++i) {
    int dim = attrs->axis.value()[i]->value;
    if (dim < 0) {
      dim = ndim + dim;
    }
    if (dim < 0 || dim >= ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Reduction operator expects all input axis indices to be in range ["
                         << -ndim << ", " << ndim << "). However, the axis index " << i << " is "
                         << attrs->axis.value()[i]->value << ", which is out of range");
    }
    if (appeared_axes[dim]) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Reduction operator expects all input axis indices not to duplicate. "
                            "However, there are at least two indices correspond to dim "
                         << dim);
    }
    appeared_axes[dim] = true;
  }

  return DynTensorType(attrs->keepdims ? ndim : ndim - n_axis, type->dtype);
}

/* relax.sum */
RELAX_REGISTER_REDUCTION_OP("sum");

/* relax.mean */
RELAX_REGISTER_REDUCTION_OP("mean");

/* relax.variance */
RELAX_REGISTER_REDUCTION_OP("variance");

/* relax.max */
RELAX_REGISTER_REDUCTION_OP("max");

}  // namespace relax
}  // namespace tvm
