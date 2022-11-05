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

#ifndef TVM_RELAX_OP_TRAINING_TRAINING_TENSOR_H_
#define TVM_RELAX_OP_TRAINING_TRAINING_TENSOR_H_

#include "training.h"

namespace tvm {
namespace relax {

Optional<Expr> InferShapeTranspose(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Unary op should have 1 argument");
  }
  Expr shape = call->args[0]->shape();
  if (auto* s = shape.as<ShapeExprNode>()) {
    return ShapeExpr(Array<PrimExpr>{s->values[1], s->values[0]});
  } else {
    return NullOpt;
  }
}

Optional<Expr> InferShapeSum(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Sum op should have 1 arguments");
  }
  Expr shape = call->args[0]->shape();
  auto* s = shape.as<ShapeExprNode>();
  if (s) {
    return ShapeExpr(Array<PrimExpr>{});
  } else {
    return NullOpt;
  }
}

Type InferTypeSum(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Sum op should have 1 arguments");
  }
  Type type0 = call->args[0]->checked_type();
  auto* t0 = type0.as<DynTensorTypeNode>();
  DataType output_dtype;
  if (t0->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else {
    output_dtype = t0->dtype;
  }
  int output_ndim = 0;
  return DynTensorType(output_ndim, output_dtype);
}

Optional<Expr> InferShapeFull(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Full op should have 1 arguments");
  }
  return call->args[0];
}

Type InferTypeFull(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Full op should have 1 arguments");
  }
  Type type0 = call->args[0]->checked_type();
  auto* t0 = type0.as<ShapeTypeNode>();
  if (!t0) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Full op should take a ShapeExpr");
  }
  return DynTensorType(call->args[0].as<ShapeExprNode>()->values.size(), DataType::Float(32));
}

Optional<Expr> InferShapeCollapseSumLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Matmul op should have 2 arguments");
  }
  return call->args[1]->shape();
}

Type InferTypeCollapseSumLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Matmul op should have 2 arguments");
  }
  return call->args[1]->checked_type();
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TRAINING_TRAINING_TENSOR_H_