/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.h
 * \brief shape and type deduction for binary broadcast operators.
 */

#ifndef TVM_RELAX_OP_TENSOR_BINARY_H_
#define TVM_RELAX_OP_TENSOR_BINARY_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <algorithm>
#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relax {

Optional<Expr> InferShapeBinaryBroadcast(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary broadcast op should have 2 arguments");
  }
  Expr lhs_shape = call->args[0]->shape();
  Expr rhs_shape = call->args[1]->shape();
  auto* s0 = lhs_shape.as<ShapeExprNode>();
  auto* s1 = rhs_shape.as<ShapeExprNode>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    size_t i = 1;
    for (; i <= std::min(ndim0, ndim1); ++i) {
      PrimExpr dim0 = s0->values[ndim0 - i];
      PrimExpr dim1 = s1->values[ndim1 - i];
      if (EqualConstInt(dim0, 1)) {
        output_shape.push_back(dim1);
      } else if (EqualConstInt(dim1, 1)) {
        output_shape.push_back(dim0);
      } else if (EqualCheck(dim0, dim1)) {
        output_shape.push_back(dim0);
      } else {
        // defer the computation of output shapes to runtime
        // e.g., broadcast Tensor([m, n]), Tensor([k]) -> defer to runtime
        return Call(ExternFunc(String("vm.binary_broadcast_shape_infer")),
                    {call->args[0], call->args[1]}, {}, {});
      }
    }
    size_t max_ndim = std::max(ndim0, ndim1);
    auto& longer_shape = (ndim0 > ndim1) ? s0 : s1;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape->values[max_ndim - i]);
    }
    return ShapeExpr(Array<PrimExpr>(output_shape.rbegin(), output_shape.rend()));
  } else {
    return NullOpt;
  }
}

Type InferTypeBinaryBroadcast(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary broadcast op should have 2 arguments");
  }
  Type lhs_type = call->args[0]->checked_type();
  Type rhs_type = call->args[1]->checked_type();
  auto* t0 = lhs_type.as<DynTensorTypeNode>();
  auto* t1 = rhs_type.as<DynTensorTypeNode>();
  if (!t0 || !t1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Both lhs and rhs should be DynTensor for broadcasting, but got "
                       << lhs_type->GetTypeKey() << " and " << rhs_type->GetTypeKey());
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Data types " << t0->dtype << " and " << t1->dtype
                       << " must be equal for broadcasting operators");
  } else {
    output_dtype = t0->dtype;
  }

  int output_ndim;
  if (t0->IsUnknownNdim() || t1->IsUnknownNdim()) {
    output_ndim = -1;
  } else {
    output_ndim = std::max(t0->ndim, t1->ndim);
  }
  return DynTensorType(output_ndim, output_dtype);
}

Optional<Expr> InferShapeDense(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Dense op should have 2 arguments");
  }
  Expr shape0 = call->args[0]->shape();
  Expr shape1 = call->args[1]->shape();
  auto* s0 = shape0.as<ShapeExprNode>();
  auto* s1 = shape1.as<ShapeExprNode>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    if (ndim0 != 2 || ndim1 != 2) {
      LOG(INFO) << ndim0;
      LOG(INFO) << ndim1;
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The 2 arguments of Dense must be 2D Tensors");
    }
    if (!EqualCheck(s0->values[1], s1->values[1])) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The 2 arguments of Dense must have the same number of columns");
    }
    return ShapeExpr(Array<PrimExpr>{s0->values[0], s1->values[0]});
  } else {
    return NullOpt;
  }
}

Type InferTypeDense(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Dense op should have 2 arguments");
  }
  Type type0 = call->args[0]->checked_type();
  Type type1 = call->args[1]->checked_type();
  auto* t0 = type0.as<DynTensorTypeNode>();
  auto* t1 = type1.as<DynTensorTypeNode>();
  if (!t0 || !t1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The 2 arguments of Dense should be DynTensor");
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Data types " << t0->dtype << ", and"
                                                     << t1->dtype << " must be equal for Dense");
  } else {
    output_dtype = t0->dtype;
  }

  int output_ndim;
  if (t0->IsUnknownNdim() || t1->IsUnknownNdim()) {
    output_ndim = -1;
  } else {
    output_ndim = t0->ndim;
  }
  return DynTensorType(output_ndim, output_dtype);
}

Optional<Expr> InferShapeConv2d(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Conv2d op should have 2 arguments");
  }
  Expr shape0 = call->args[0]->shape();
  Expr shape1 = call->args[1]->shape();
  auto* s0 = shape0.as<ShapeExprNode>();
  auto* s1 = shape1.as<ShapeExprNode>();
  auto* attrs = call->attrs.as<Conv2dAttrs>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    if (ndim0 != 4 || ndim1 != 4) {
      LOG(INFO) << ndim0;
      LOG(INFO) << ndim1;
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The 2 arguments of Dense must be 4D Tensors");
    }
    // N
    output_shape.push_back(s0->values[0]);
    // C
    output_shape.push_back(s1->values[0]);
    // H
    output_shape.push_back((s0->values[2] + 2 * attrs->padding[0] -
                            attrs->dilation[0] * (attrs->kernel_size[0] - 1) - 1) /
                               attrs->stride[0] +
                           1);
    // W
    output_shape.push_back((s0->values[3] + 2 * attrs->padding[1] -
                            attrs->dilation[1] * (attrs->kernel_size[1] - 1) - 1) /
                               attrs->stride[1] +
                           1);
    return ShapeExpr(Array<PrimExpr>{output_shape.begin(), output_shape.end()});
  } else {
    return NullOpt;
  }
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_BINARY_H_
