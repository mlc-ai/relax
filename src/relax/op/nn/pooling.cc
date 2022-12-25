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

#include <unordered_map>

#include "../op_common.h"

namespace tvm {
namespace relax {

/* relax.nn.max_pool2d */
TVM_REGISTER_NODE_TYPE(MaxPool2DAttrs);

Expr MakeMaxPool2D(Expr data, Array<PrimExpr> pool_size, Array<PrimExpr> strides,
                   Array<PrimExpr> padding, Array<PrimExpr> dilation, String layout,
                   Optional<String> out_layout) {
  padding = GetCompletePadding2D(std::move(padding));
  if (pool_size.size() == 1) {
    pool_size.push_back(pool_size[0]);
  }
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  CHECK_EQ(pool_size.size(), 2)
      << "The input pool_size length is expected to be 2. However, the given pool_size is "
      << pool_size;
  CHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;
  CHECK(CheckTensorLayout(layout, {'N', 'C', 'H', 'W'}))
      << "The input data layout is expected to exactly contain \"N\", \"C\", \"H\" and \"W\". "
         "However, the given data layout is "
      << layout;
  CHECK(!out_layout.defined() || CheckTensorLayout(out_layout.value(), {'N', 'C', 'H', 'W'}))
      << "The given output layout is expected to exactly contain \"N\", \"C\", \"H\" and \"W\". "
         "However, the given output layout is "
      << out_layout.value();

  auto attrs = make_object<MaxPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  static const Op& op = Op::Get("relax.nn.max_pool2d");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.max_pool2d").set_body_typed(MakeMaxPool2D);

StructInfo InferStructInfoMaxPool2D(const Call& call, const BlockBuilder& ctx) {
  constexpr static int ndim = 4;
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx, /*op_name=*/"MaxPool2D");

  Optional<ShapeExpr> data_shape =
      CheckNdimAndGetShape(call, ctx, data_sinfo, ndim, /*op_name=*/"MaxPool2D");
  if (!data_shape.defined()) {
    return TensorStructInfo(data_sinfo->dtype, ndim);
  }

  const auto* attrs = call->attrs.as<MaxPool2DAttrs>();
  ShapeExpr _data_shape = data_shape.value();
  std::string _data_layout = attrs->layout.operator std::string();

  std::unordered_map<char, PrimExpr> output_shape;
  PrimExpr batch = _data_shape->values[_data_layout.find('N')];
  PrimExpr channel = _data_shape->values[_data_layout.find('C')];
  PrimExpr input_h = _data_shape->values[_data_layout.find('H')];
  PrimExpr input_w = _data_shape->values[_data_layout.find('W')];
  PrimExpr kernel_h = attrs->pool_size[0];
  PrimExpr kernel_w = attrs->pool_size[1];
  PrimExpr padding_h = attrs->padding[0] + attrs->padding[2];
  PrimExpr padding_w = attrs->padding[1] + attrs->padding[3];

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  output_shape['N'] = batch;
  output_shape['C'] = channel;
  output_shape['H'] = analyzer->Simplify(
      (input_h + padding_h - attrs->dilation[0] * (kernel_h - 1) - 1) / attrs->strides[0] + 1);
  output_shape['W'] = analyzer->Simplify(
      (input_w + padding_w - attrs->dilation[1] * (kernel_w - 1) - 1) / attrs->strides[1] + 1);

  std::vector<PrimExpr> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(output_shape[attrs->out_layout.at(i)]);
  }
  return TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.nn.max_pool2d")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<MaxPool2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMaxPool2D);

/* relax.nn.adaptive_avg_pool2d */
TVM_REGISTER_NODE_TYPE(AdaptivePool2DAttrs);

Expr MakeAdaptiveAvgPool2D(Expr data, Optional<Array<PrimExpr>> output_size, String layout,
                           Optional<String> out_layout) {
  CHECK(CheckTensorLayout(layout, {'N', 'C', 'H', 'W'}))
      << "The input data layout is expected to exactly contain \"N\", \"C\", \"H\" and \"W\". "
         "However, the given data layout is "
      << layout;
  CHECK(!out_layout.defined() || CheckTensorLayout(out_layout.value(), {'N', 'C', 'H', 'W'}))
      << "The given output layout is expected to exactly contain \"N\", \"C\", \"H\" and \"W\". "
         "However, the given output layout is "
      << out_layout.value();

  ObjectPtr<AdaptivePool2DAttrs> attrs = make_object<AdaptivePool2DAttrs>();
  attrs->layout = layout;
  attrs->out_layout = out_layout.value_or(layout);
  if (output_size.defined()) {
    Array<PrimExpr> _output_size = output_size.value();
    if (_output_size.size() == 1) {
      _output_size.push_back(_output_size[0]);
    }
    CHECK_EQ(_output_size.size(), 2)
        << "The output_size length is expected to be 2. However, the given output_size is "
        << _output_size;
    attrs->output_size = std::move(_output_size);
  }

  static const Op& op = Op::Get("relax.nn.adaptive_avg_pool2d");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.adaptive_avg_pool2d").set_body_typed(MakeAdaptiveAvgPool2D);

StructInfo InferStructInfoAdaptiveAvgPool2D(const Call& call, const BlockBuilder& ctx) {
  constexpr static int ndim = 4;
  TensorStructInfo data_sinfo =
      GetUnaryInputTensorStructInfo(call, ctx, /*op_name=*/"AdaptiveAvgPool2D");

  Optional<ShapeExpr> data_shape =
      CheckNdimAndGetShape(call, ctx, data_sinfo, ndim, /*op_name=*/"AdaptiveAvgPool2D");
  if (!data_shape.defined()) {
    return TensorStructInfo(data_sinfo->dtype, ndim);
  }

  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
  ShapeExpr _data_shape = data_shape.value();
  std::string _data_layout = attrs->layout.operator std::string();

  std::unordered_map<char, PrimExpr> output_shape;
  output_shape['N'] = _data_shape->values[_data_layout.find('N')];
  output_shape['C'] = _data_shape->values[_data_layout.find('C')];
  if (attrs->output_size.defined()) {
    output_shape['H'] = attrs->output_size.value()[0];
    output_shape['W'] = attrs->output_size.value()[1];
  } else {
    output_shape['H'] = _data_shape->values[_data_layout.find('H')];
    output_shape['W'] = _data_shape->values[_data_layout.find('W')];
  }

  std::vector<PrimExpr> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(output_shape[attrs->out_layout.at(i)]);
  }
  return TensorStructInfo(ShapeExpr(shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.nn.adaptive_avg_pool2d")
    .set_attrs_type<AdaptivePool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAdaptiveAvgPool2D);

}  // namespace relax
}  // namespace tvm
