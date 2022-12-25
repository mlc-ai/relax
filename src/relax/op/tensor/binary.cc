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
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include <algorithm>
#include <utility>
#include <vector>

#include "../op_common.h"

namespace tvm {
namespace relax {

#define RELAX_REGISTER_BINARY_OP(OpName)                                          \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr lhs, Expr rhs) { \
    static const Op& op = Op::Get("relax." OpName);                               \
    return Call(op, {lhs, rhs}, Attrs(), {});                                     \
  });                                                                             \
  RELAX_REGISTER_OP("relax." OpName)                                              \
      .set_num_inputs(2)                                                          \
      .add_argument("lhs", "Tensor", "The left hand side tensor.")                \
      .add_argument("rhs", "Tensor", "The right hand side tensor.")

#define RELAX_REGISTER_BINARY_BROADCAST_OP(OpName)                                \
  RELAX_REGISTER_BINARY_OP(OpName).set_attr<FInferStructInfo>("FInferStructInfo", \
                                                              InferStructInfoBroadcast)

#define RELAX_REGISTER_CMP_OP(OpName)                                             \
  RELAX_REGISTER_BINARY_OP(OpName).set_attr<FInferStructInfo>("FInferStructInfo", \
                                                              InferStructInfoCMP)

using FComputeOutDtype = std::function<DataType(
    const Call&, const BlockBuilder&, const TensorStructInfoNode*, const TensorStructInfoNode*)>;

StructInfo InferStructInfoBinary(const Call& call, const BlockBuilder& ctx,
                                 FComputeOutDtype f_compute_out_dtype) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Binary op should have 2 arguments");
  }
  auto* lhs_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  auto* rhs_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  if (!lhs_sinfo || !rhs_sinfo) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Both lhs and rhs should be Tensor for binary operator, but got "
                     << call->args[0]->struct_info_->GetTypeKey() << " and "
                     << call->args[1]->struct_info_->GetTypeKey());
  }

  // DateType
  DataType output_dtype = f_compute_out_dtype(call, ctx, lhs_sinfo, rhs_sinfo);

  // ndims
  int output_ndim;
  if (lhs_sinfo->IsUnknownNdim() || rhs_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(lhs_sinfo->ndim, rhs_sinfo->ndim);
  }

  auto* lhs_shape = lhs_sinfo->shape.as<ShapeExprNode>();
  auto* rhs_shape = rhs_sinfo->shape.as<ShapeExprNode>();
  // Shapes and ndims
  if (lhs_shape && rhs_shape) {
    // If all inputs have shapes, directly infer shapes
    std::vector<PrimExpr> output_shape;

    size_t lhs_ndim = lhs_sinfo->ndim;
    size_t rhs_ndim = rhs_sinfo->ndim;
    size_t max_ndim = std::max(lhs_ndim, rhs_ndim);

    size_t i = 1;
    for (; i <= std::min(lhs_ndim, rhs_ndim); ++i) {
      const PrimExpr& dim0 = lhs_shape->values[lhs_ndim - i];
      const PrimExpr& dim1 = rhs_shape->values[rhs_ndim - i];
      if (EqualConstInt(dim0, 1)) {
        output_shape.push_back(dim1);
      } else if (EqualConstInt(dim1, 1)) {
        output_shape.push_back(dim0);
      } else if (EqualCheck(dim0, dim1)) {
        output_shape.push_back(dim0);
      } else {
        // Use simple fallback when shape mismatch.
        return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
      }
    }
    auto& longer_shape = (lhs_ndim > rhs_ndim) ? lhs_shape : rhs_shape;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape->values[max_ndim - i]);
    }
    Expr shape = ShapeExpr(Array<PrimExpr>(output_shape.rbegin(), output_shape.rend()));
    return TensorStructInfo(shape, output_dtype);
  } else {
    return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
  }
}

StructInfo InferStructInfoBroadcast(const Call& call, const BlockBuilder& ctx) {
  auto f_compute_out_dtype = [](const Call& call, const BlockBuilder& ctx,
                                const TensorStructInfoNode* lhs_sinfo,
                                const TensorStructInfoNode* rhs_sinfo) {
    DataType output_dtype;
    if (lhs_sinfo->IsUnknownDtype() || rhs_sinfo->IsUnknownDtype()) {
      output_dtype = DataType::Void();
    } else if (lhs_sinfo->dtype != rhs_sinfo->dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Data types " << lhs_sinfo->dtype << " and " << rhs_sinfo->dtype
                       << " must be equal for broadcasting operators");
    } else {
      output_dtype = lhs_sinfo->dtype;
    }
    return output_dtype;
  };
  return InferStructInfoBinary(call, ctx, std::move(f_compute_out_dtype));
}

StructInfo InferStructInfoCMP(const Call& call, const BlockBuilder& ctx) {
  auto f_compute_out_dtype = [](const Call& call, const BlockBuilder& ctx,
                                const TensorStructInfoNode* lhs_sinfo,
                                const TensorStructInfoNode* rhs_sinfo) { return DataType::Bool(); };
  return InferStructInfoBinary(call, ctx, std::move(f_compute_out_dtype));
}

RELAX_REGISTER_BINARY_BROADCAST_OP("add").describe("Elementwise addition with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP("subtract")
    .describe("Elementwise subtraction with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP("multiply")
    .describe("Elementwise multiplication with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP("divide").describe("Elementwise division with broadcasting");

RELAX_REGISTER_BINARY_BROADCAST_OP("floor_divide")
    .describe("Elementwise floor-division with broadcasting");

RELAX_REGISTER_CMP_OP("less");

}  // namespace relax
}  // namespace tvm
