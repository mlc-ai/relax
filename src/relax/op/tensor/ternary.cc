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

#include "../op_common.h"

/*!
 * \file ternary.cc
 * \brief ternary operators.
 */

namespace tvm {
namespace relax {

StructInfo InferStructInfoEwiseFMA(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo t0 = input_sinfo[0];
  TensorStructInfo t1 = input_sinfo[1];
  TensorStructInfo t2 = input_sinfo[2];

  int ndim = kUnknownNDim;
  if (!t0->IsUnknownNdim()) {
    ndim = t0->ndim;
  }
  if (!t1->IsUnknownNdim()) {
    if (ndim == kUnknownNDim) {
      ndim = t1->ndim;
    } else if (t1->ndim != ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The 3 arguments of EwiseFMA must have the same number of dimensions");
    }
  }
  if (!t2->IsUnknownNdim()) {
    if (ndim == kUnknownNDim) {
      ndim = t2->ndim;
    } else if (t2->ndim != ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "The 3 arguments of EwiseFMA must have the same number of dimensions");
    }
  }

  DataType output_dtype;
  if (t0->IsUnknownDtype() || t1->IsUnknownDtype() || t2->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (t0->dtype != t1->dtype || t1->dtype != t2->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << t0->dtype << ", " << t1->dtype << ", and " << t2->dtype
                     << " must be equal for EwiseFMA");
  } else {
    output_dtype = t0->dtype;
  }

  auto* s0 = t0->shape.as<ShapeExprNode>();
  auto* s1 = t1->shape.as<ShapeExprNode>();
  auto* s2 = t2->shape.as<ShapeExprNode>();
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  if (s0 && s1 && s2) {
    Array<PrimExpr> output_shape;
    for (int i = 0; i < ndim; ++i) {
      PrimExpr dim0 = s0->values[i];
      PrimExpr dim1 = s1->values[i];
      PrimExpr dim2 = s2->values[i];
      if (analyzer->CanProveEqual(dim0, dim1) && analyzer->CanProveEqual(dim1, dim2)) {
        output_shape.push_back(dim0);
      } else {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "The 3 arguments of EwiseFMA must have the same shape");
      }
    }
    return TensorStructInfo(ShapeExpr(output_shape), output_dtype);
  } else if (t0->shape.defined() && t0->shape.same_as(t1->shape) && t0->shape.same_as(t2->shape)) {
    return TensorStructInfo(t0->shape.value(), output_dtype);
  }

  return TensorStructInfo(output_dtype, ndim);
}

TVM_REGISTER_OP("relax.ewise_fma")
    .set_num_inputs(3)
    .add_argument("e0", "Tensor", "The left hand operand of the multiplication")
    .add_argument("e1", "Tensor", "The right hand operand of the multiplication")
    .add_argument("e2", "Tensor", "The operand of the addition")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoEwiseFMA);

Expr MakeEwiseFma(Expr expr0, Expr expr1, Expr expr2) {
  static const Op& op = Op::Get("relax.ewise_fma");
  return Call(op, {expr0, expr1, expr2}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.ewise_fma").set_body_typed(MakeEwiseFma);

}  // namespace relax
}  // namespace tvm
