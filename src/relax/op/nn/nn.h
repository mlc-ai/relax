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

#ifndef TVM_RELAX_OP_NN_NN_H_
#define TVM_RELAX_OP_NN_NN_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include "../op_common.h"
#include "../tensor/unary.h"
namespace tvm {
namespace relax {

Optional<Expr> InferShapeFlatten(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Flatten op should have 1 argument");
  }
  Expr shape = call->args[0]->shape();
  auto* s = shape.as<ShapeExprNode>();
  if (s) {
    PrimExpr output_dim = 1;
    for (int i = 1; i < static_cast<int>(s->values.size()); i++) {
      output_dim *= s->values[i];
    }
    return ShapeExpr({s->values[0], output_dim});
  } else {
    return NullOpt;
  }
}

Type InferTypeFlatten(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Flatten op should have 1 argument");
  }
  auto* input_ty = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (!input_ty) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Input should be DynTensor, but got "
                       << call->args[0]->checked_type()->GetTypeKey());
  }
  return DynTensorType(/*ndim=*/2, input_ty->dtype);
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

}  // namespace relax
}  // namespace tvm
#endif