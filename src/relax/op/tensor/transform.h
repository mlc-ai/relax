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
 * \file src/relax/op/tensor/transform.h
 * \brief Transform op attributes.
 */

#ifndef TVM_RELAX_OP_TENSOR_TRANSFORM_H_
#define TVM_RELAX_OP_TENSOR_TRANSFORM_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/* relax.transpose */
Expr InferShapeTranspose(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeTranspose(const Call& call, DiagnosticContext diag_ctx);

/* relax.reshape */
Expr InferShapeReshape(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeReshape(const Call& call, DiagnosticContext diag_ctx);

/* relax.expand_dims */
Expr InferShapeExpandDims(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeExpandDims(const Call& call, DiagnosticContext diag_ctx);

/* relax.squeeze */
Expr InferShapeSqueeze(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeSqueeze(const Call& call, DiagnosticContext diag_ctx);

/* relax.concatenate */
Expr InferShapeConcatenate(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeConcatenate(const Call& call, DiagnosticContext diag_ctx);

/* relax.cumsum */
Expr InferShapeCumsum(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeCumsum(const Call& call, DiagnosticContext diag_ctx);

/* relax.trilu */
Expr InferShapeTrilu(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeTrilu(const Call& call, DiagnosticContext diag_ctx);

/* relax.cast */
Expr InferShapeCast(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeCast(const Call& call, DiagnosticContext diag_ctx);

/* relax.wrap_param */
Expr InferShapeWrapParam(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeWrapParam(const Call& call, DiagnosticContext diag_ctx);

/* relax.take */
Expr InferShapeTake(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeTake(const Call& call, DiagnosticContext diag_ctx);

/* relax.full */
Expr InferShapeFull(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeFull(const Call& call, DiagnosticContext diag_ctx);

/* relax.split */
Expr InferShapeSplit(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeSplit(const Call& call, DiagnosticContext diag_ctx);

/* relax.broadcast_to */
Expr InferShapeBroadcastTo(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeBroadcastTo(const Call& call, DiagnosticContext diag_ctx);

/* relax.strided_slice */
Expr InferShapeStridedSlice(const Call& call, DiagnosticContext diag_ctx);

Type InferTypeStridedSlice(const Call& call, DiagnosticContext diag_ctx);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_TRANSFORM_H_