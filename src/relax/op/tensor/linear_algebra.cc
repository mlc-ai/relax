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
 * \file linear_algebra.cc
 * \brief Linear algebra operators.
 */

#include "linear_algebra.h"

#include <algorithm>
#include <utility>

namespace tvm {
namespace relax {

/* relax.matmul */
TVM_REGISTER_NODE_TYPE(MatmulAttrs);

Expr Matmul(Expr a, Expr b, DataType out_dtype) {
  ObjectPtr<MatmulAttrs> attrs = make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.matmul");
  return Call(op, {std::move(a), std::move(b)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.matmul").set_body_typed(Matmul);

StructInfo InferStructInfoMatmul(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo lhs_sinfo = input_sinfo[0];
  TensorStructInfo rhs_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<MatmulAttrs>();
  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, lhs_sinfo, rhs_sinfo)
                           : attrs->out_dtype;

  if (lhs_sinfo->IsUnknownNdim() || rhs_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(out_dtype, kUnknownNDim);
  }
  int lhs_ndim = lhs_sinfo->ndim;
  int rhs_ndim = rhs_sinfo->ndim;
  if (lhs_ndim == 0 || rhs_ndim == 0) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Matmul requires both inputs to have at least 1 dimension. However, "
                     << (lhs_ndim == 0 ? "lhs" : "rhs") << " is a 0-rank tensor.");
  }

  int lhs_prepended = 0;
  int rhs_appended = 0;
  if (lhs_ndim == 1) {
    lhs_ndim = 2;
    lhs_prepended = 1;
  }
  if (rhs_ndim == 1) {
    rhs_ndim = 2;
    rhs_appended = 1;
  }
  int output_ndim = std::max(lhs_ndim, rhs_ndim) - lhs_prepended - rhs_appended;

  const auto* lhs_shape = lhs_sinfo->shape.as<ShapeExprNode>();
  const auto* rhs_shape = rhs_sinfo->shape.as<ShapeExprNode>();
  if (lhs_shape == nullptr || rhs_shape == nullptr) {
    return TensorStructInfo(out_dtype, output_ndim);
  }

  Array<PrimExpr> lhs_shape_prefix{lhs_shape->values.begin(),
                                   lhs_shape->values.end() - 2 + lhs_prepended};
  Array<PrimExpr> rhs_shape_prefix{rhs_shape->values.begin(),
                                   rhs_shape->values.end() - 2 + rhs_appended};
  Optional<Array<PrimExpr>> output_shape_prefix =
      InferBinaryBroadcastShape(call, ctx, lhs_shape_prefix, rhs_shape_prefix);
  if (!output_shape_prefix.defined()) {
    return TensorStructInfo(out_dtype, output_ndim);
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr lhs_reduction_length = lhs_shape->values[lhs_sinfo->ndim - 1];
  PrimExpr rhs_reduction_length = rhs_shape->values[rhs_ndim - 2];
  if (analyzer->CanProve(lhs_reduction_length != rhs_reduction_length)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Matmul requires the reduction length of lhs and rhs to be equal. However, "
                        "the reduction lengths of lhs and rhs are "
                     << lhs_reduction_length << " and " << rhs_reduction_length
                     << " respectively.");
  }

  Array<PrimExpr> output_shape = output_shape_prefix.value();
  if (!lhs_prepended) {
    output_shape.push_back(lhs_shape->values[lhs_ndim - 2]);
  }
  if (!rhs_appended) {
    output_shape.push_back(rhs_shape->values[rhs_ndim - 1]);
  }
  ICHECK_EQ(static_cast<int>(output_shape.size()), output_ndim);
  return TensorStructInfo(ShapeExpr(output_shape), out_dtype);
}

TVM_REGISTER_OP("relax.matmul")
    .set_num_inputs(2)
    .add_argument("a", "Tensor", "The left operand of the matmul.")
    .add_argument("b", "Tensor", "The right operand of the matmul.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMatmul);

}  // namespace relax
}  // namespace tvm
