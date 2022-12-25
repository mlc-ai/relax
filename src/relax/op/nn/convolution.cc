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

#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {

/* relax.nn.conv2d */
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

Expr MakeConv2D(Expr data, Expr weight, Array<PrimExpr> strides, Array<PrimExpr> padding,
                Array<PrimExpr> dilation, String data_layout, String kernel_layout,
                Optional<String> out_layout, DataType out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  CHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;
  CHECK(CheckTensorLayout(data_layout, {'N', 'C', 'H', 'W'}))
      << "The input data layout is expected to exactly contain \"N\", \"C\", \"H\" and \"W\". "
         "However, the given data layout is "
      << data_layout;
  CHECK(CheckTensorLayout(kernel_layout, {'I', 'O', 'H', 'W'}))
      << "The input kernel layout is expected to exactly contain \"I\", \"O\", \"H\" and \"W\". "
         "However, the given kernel layout is "
      << data_layout;
  CHECK(!out_layout.defined() || CheckTensorLayout(out_layout.value(), {'N', 'C', 'H', 'W'}))
      << "The given output layout is expected to exactly contain \"N\", \"C\", \"H\" and \"W\". "
         "However, the given output layout is "
      << out_layout.value();
  return MakeConv<Conv2DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype, /*op_name=*/"relax.nn.conv2d");
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv2d").set_body_typed(MakeConv2D);

StructInfo InferStructInfoConv2d(const Call& call, const BlockBuilder& ctx) {
  constexpr static int ndim = 4;
  auto [data_sinfo, weight_sinfo] = GetBinaryInputTensorStructInfo(call, ctx, /*op_name=*/"Conv2D");

  Optional<ShapeExpr> data_shape =
      CheckNdimAndGetShape(call, ctx, data_sinfo, ndim, /*op_name=*/"Conv2D");
  Optional<ShapeExpr> weight_shape =
      CheckNdimAndGetShape(call, ctx, weight_sinfo, ndim, /*op_name=*/"Conv2D");
  const auto* attrs = call->attrs.as<Conv2DAttrs>();

  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, data_sinfo, weight_sinfo)
                           : attrs->out_dtype;
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, ndim);
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  ShapeExpr _data_shape = data_shape.value();
  ShapeExpr _weight_shape = weight_shape.value();
  std::string _data_layout = attrs->data_layout.operator std::string();
  std::string _kernel_layout = attrs->kernel_layout.operator std::string();

  PrimExpr input_channel_data = _data_shape->values[_data_layout.find('C')];
  PrimExpr input_channel_kernel = _weight_shape->values[_kernel_layout.find('I')];
  if (analyzer->CanProve(input_channel_data - input_channel_kernel != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call->span)
                     << "The channel size of the data should equal to the input channel size of "
                        "the weight. However, the data channel size is "
                     << input_channel_data << " while the weight input channel size is "
                     << input_channel_kernel);
  }

  std::unordered_map<char, PrimExpr> output_shape;
  PrimExpr batch = _data_shape->values[_data_layout.find('N')];
  PrimExpr output_channel = _weight_shape->values[_kernel_layout.find('O')];
  PrimExpr input_h = _data_shape->values[_data_layout.find('H')];
  PrimExpr input_w = _data_shape->values[_data_layout.find('W')];
  PrimExpr kernel_h = _weight_shape->values[_kernel_layout.find('H')];
  PrimExpr kernel_w = _weight_shape->values[_kernel_layout.find('W')];
  PrimExpr padding_h = attrs->padding[0] + attrs->padding[2];
  PrimExpr padding_w = attrs->padding[1] + attrs->padding[3];
  output_shape['N'] = batch;
  output_shape['C'] = output_channel;
  output_shape['H'] = analyzer->Simplify(
      (input_h + padding_h - attrs->dilation[0] * (kernel_h - 1) - 1) / attrs->strides[0] + 1);
  output_shape['W'] = analyzer->Simplify(
      (input_w + padding_w - attrs->dilation[1] * (kernel_w - 1) - 1) / attrs->strides[1] + 1);

  std::vector<PrimExpr> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(output_shape[attrs->out_layout.at(i)]);
  }
  return TensorStructInfo(ShapeExpr(shape), out_dtype);
}

TVM_REGISTER_OP("relax.nn.conv2d")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2d);

}  // namespace relax
}  // namespace tvm
