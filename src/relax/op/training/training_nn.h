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

#ifndef TVM_RELAX_OP_TRAINING_TRAINING_NN_H_
#define TVM_RELAX_OP_TRAINING_TRAINING_NN_H_

#include "training.h"

namespace tvm {
namespace relax {

/* Infer Shape & Type */
Optional<Expr> InferShapeCrossEntropy(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "CrossEnt op should have 2 arguments");
  }
  Expr shape0 = call->args[0]->shape();
  Expr shape1 = call->args[1]->shape();
  auto* s0 = shape0.as<ShapeExprNode>();
  auto* s1 = shape1.as<ShapeExprNode>();
  if (s0 && s1) {
    return ShapeExpr(Array<PrimExpr>{});
  } else {
    return NullOpt;
  }
}

Type InferTypeCrossEntropy(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "CrossEntropy op should have 2 arguments");
  }
  Type type0 = call->args[0]->checked_type();
  Type type1 = call->args[1]->checked_type();
  auto* t0 = type0.as<DynTensorTypeNode>();
  auto* t1 = type1.as<DynTensorTypeNode>();
  if (!t0 || !t1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The 2 arguments of CrossEntropy should be DynTensor");
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Data types " << t0->dtype << ", and"
                                                     << t1->dtype << " must be equal for CrossEntropy");
  } else {
    output_dtype = t0->dtype;
  }

  int output_ndim = 0;
  // if (t0->IsUnknownNdim() || t1->IsUnknownNdim()) {
  //   output_ndim = -1;
  // } else {
  //   output_ndim = t0->ndim;
  // }
  return DynTensorType(output_ndim, output_dtype);
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TRAINING_TRAINING_NN_H_
