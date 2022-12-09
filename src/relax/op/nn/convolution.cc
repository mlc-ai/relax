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
 * \file src/relax/op/nn/convolution.cc
 * \brief Convolution operators
 */

#include "convolution.h"

#include "../tensor/binary.h"
namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

Expr MakeConv2D(Expr data, Expr weight, Array<PrimExpr> strides, Array<PrimExpr> padding,
                Array<PrimExpr> dilation, int groups, PrimExpr channels,
                Array<PrimExpr> kernel_size, String data_layout, String kernel_layout,
                String out_layout, DataType out_dtype) {
  return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                               kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                               "relax.nn.conv2d");
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv2d").set_body_typed(MakeConv2D);

RELAX_REGISTER_OP("relax.nn.conv2d")
    .describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeConv2D)
    .set_attr<FInferType>("FInferType", InferTypeConv2D);

Expr InferShapeConv2D(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Conv2d op should have 2 arguments");
  }
  Expr shape0 = call->args[0]->shape();
  Expr shape1 = call->args[1]->shape();
  auto* s0 = shape0.as<ShapeExprNode>();
  auto* s1 = shape1.as<ShapeExprNode>();
  auto* attrs = call->attrs.as<Conv2DAttrs>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    if (ndim0 != 4 || ndim1 != 4) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The 2 arguments of Conv2d must be 4D Tensors");
    }
    // N
    output_shape.push_back(s0->values[0]);
    // C
    output_shape.push_back(s1->values[0]);
    // H
    output_shape.push_back((s0->values[2] + 2 * attrs->padding[0] -
                            attrs->dilation[0] * (attrs->kernel_size[0] - 1) - 1) /
                               attrs->strides[0] +
                           1);
    // W
    output_shape.push_back((s0->values[3] + 2 * attrs->padding[1] -
                            attrs->dilation[1] * (attrs->kernel_size[1] - 1) - 1) /
                               attrs->strides[1] +
                           1);
    return ShapeExpr(Array<PrimExpr>{output_shape.begin(), output_shape.end()});
  } else {
    return RuntimeDepShape();
  }
}

Type InferTypeConv2D(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Conv2d op should have 2 arguments");
  }

  const auto* data_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* kernel_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  if (data_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }
  if (kernel_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input kernel should has type DynTensorType, but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }

  if (!data_type->IsUnknownNdim() && data_type->ndim != 4) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The operator conv2d expects the input data to be a 4-rank tensor. "
                          "However, the given data has rank "
                       << data_type->ndim);
  }
  if (!kernel_type->IsUnknownNdim() && kernel_type->ndim != 4) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The operator conv2d expects the input kernel to be a 4-rank tensor. "
                          "However, the given kernel has rank "
                       << kernel_type->ndim);
  }

  DataType output_dtype;
  if (data_type->IsUnknownDtype() || kernel_type->IsUnknownDtype()) {
    output_dtype = attrs->out_dtype;
  } else if (data_type->dtype != kernel_type->dtype && attrs->out_dtype.is_void()) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "Conv2D expects both the input data and kernel to have the same data type when there is "
           "no specified output dtype. However, the given data has dtype "
        << data_type->dtype << " while the given kernel has dtype " << kernel_type->dtype);
  } else {
    output_dtype = attrs->out_dtype.is_void() ? data_type->dtype : attrs->out_dtype;
  }
  return DynTensorType(4, output_dtype);
}

}  // namespace relax
}  // namespace tvm
