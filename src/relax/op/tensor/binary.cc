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

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Quick helper macro
 * - Expose a make function to construct the node.
 * - Register op to the registry.
 * \param OpName The name of operator to register. The name passed in will
 *  1. be prepended with a prefix "relax.op." as the FFI key string for the make function,
 *  2. be prepended with a prefix "relax." as the key string in the operator registry.
 */
#define RELAX_REGISTER_BINARY_OP(OpName)                                          \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr lhs, Expr rhs) { \
    static const Op& op = Op::Get("relax." OpName);                               \
    return Call(op, {lhs, rhs}, Attrs(), {});                                     \
  });                                                                             \
  TVM_REGISTER_OP("relax." OpName)                                                \
      .set_num_inputs(2)                                                          \
      .add_argument("lhs", "Tensor", "The left hand side tensor.")                \
      .add_argument("rhs", "Tensor", "The right hand side tensor.")

#define RELAX_REGISTER_BINARY_BROADCAST_OP(OpName)                                \
  RELAX_REGISTER_BINARY_OP(OpName).set_attr<FInferStructInfo>("FInferStructInfo", \
                                                              InferStructInfoBroadcastArith)

#define RELAX_REGISTER_CMP_OP(OpName)                                             \
  RELAX_REGISTER_BINARY_OP(OpName).set_attr<FInferStructInfo>("FInferStructInfo", \
                                                              InferStructInfoBroadcastCMP)

template <typename FType>
StructInfo InferStructInfoBroadcast(const Call& call, const BlockBuilder& ctx,
                                    FType f_compute_out_dtype) {
  auto [lhs_sinfo, rhs_sinfo] = GetBinaryInputTensorStructInfo(call, ctx, /*op_name=*/"Binary");

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
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
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
      const auto* int_dim0 = dim0.as<IntImmNode>();
      const auto* int_dim1 = dim1.as<IntImmNode>();
      if (int_dim0 != nullptr && int_dim0->value == 1) {
        output_shape.push_back(dim1);
      } else if (int_dim1 != nullptr && int_dim1->value == 1) {
        output_shape.push_back(dim0);
      } else if (analyzer->CanProveEqual(dim0, dim1)) {
        output_shape.push_back(dim0);
      } else if (int_dim0 && int_dim1 && int_dim0->value != int_dim1->value) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "The lhs shape at dim " << lhs_ndim - i << " is " << dim0
                         << " and the rhs shape at dim " << rhs_ndim - i << " is " << dim1
                         << ", which are not broadcastable.");
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

StructInfo InferStructInfoBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(call, ctx, InferBinaryArithOpOutDtype);
}

StructInfo InferStructInfoBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorStructInfo& lhs_sinfo,
         const TensorStructInfo& rhs_sinfo) { return DataType::Bool(); });
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
