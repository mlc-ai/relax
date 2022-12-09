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

#include "pooling.h"

#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/nn/pooling.h>

#include "../tensor/unary.h"
namespace tvm {
namespace relax {

/* relax.nn.max_pool2d */
TVM_REGISTER_NODE_TYPE(MaxPool2DAttrs);

template <typename AttrType, topi::nn::PoolType mode>
Array<te::Tensor> Pool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  static const tir::Layout kNCHW("NCHW");
  const auto* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto dilation = param->dilation;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  tir::Layout layout(param->layout);
  tir::Layout out_layout(param->out_layout);

  ICHECK(tir::BijectiveLayout(layout, kNCHW).defined())
      << "max_pool2d currently only supports layouts that are convertible from NCHW";
  ICHECK_EQ(layout.IndexOf(tir::LayoutAxis::Get('h')), -1)
      << "max_pool2d does not support input split on height";
  ICHECK_EQ(layout.IndexOf(tir::LayoutAxis::Get('w')), -1)
      << "max_pool2d does not support input split on width";

  ICHECK(inputs[0].ndim() == 4U || inputs[0].ndim() == 5U || inputs[0].ndim() == 6U)
      << "Pool2D only support 4-D input (e.g., NCHW)"
      << " or 5-D input (e.g. NCHWc on for vector instructions)"
      << " or 6-D input (e.g. NCHWnc for tensor accelerators)";

  if (param->padding.size() == 1) {
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
  } else if (param->padding.size() == 2) {
    padding.push_back(padding[0]);
    padding.push_back(padding[1]);
  }
  if (mode == topi::nn::kAvgPool) {
    // TODO(@sunggg): Disabled for now until implementing Avg Pool
    // bool count_include_pad = reinterpret_cast<const AvgPool2DAttrs*>(param)->count_include_pad;
    // return Array<te::Tensor>{topi::nn::pool2d(inputs[0], pool_size, stride, dilation, padding,
    //                                          mode, ceil_mode, layout.name(), count_include_pad)};
  } else {
    return Array<te::Tensor>{topi::nn::pool2d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name())};
  }
}

Expr InferShapeMaxPool2d(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "MaxPool2d op should have 1 argument");
  }
  auto attrs = call->attrs.as<MaxPool2DAttrs>();
  Expr shape = call->args[0]->shape();
  auto* s = shape.as<ShapeExprNode>();
  if (s) {
    Array<PrimExpr> output_shape;
    for (int i = 0; i < static_cast<int>(s->values.size()); i++) {
      if (i == static_cast<int>(s->values.size()) - 2) {
        output_shape.push_back((s->values[i] + 2 * attrs->padding[0] -
                                attrs->dilation[0] * (attrs->pool_size[0] - 1) - 1) /
                                   attrs->strides[0] +
                               1);
      } else if (i == static_cast<int>(s->values.size()) - 1) {
        output_shape.push_back((s->values[i] + 2 * attrs->padding[1] -
                                attrs->dilation[1] * (attrs->pool_size[1] - 1) - 1) /
                                   attrs->strides[1] +
                               1);
      } else {
        output_shape.push_back(s->values[i]);
      }
    }
    return ShapeExpr(Array<PrimExpr>{output_shape.begin(), output_shape.end()});
  } else {
    return RuntimeDepShape();
  }
}


RELAX_REGISTER_OP("relax.nn.max_pool2d")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<MaxPool2DAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeMaxPool2d)
    .set_attr<FInferType>("FInferType", InferTypeUnaryBroadcast)
    // TODO(@tvm-team): Can we implement this in TOPI instead?
    .set_attr<FTVMCompute>("FTVMCompute", Pool2DCompute<MaxPool2DAttrs, topi::nn::kMaxPool>);

Expr MakeMaxPool2D(Expr data, Array<PrimExpr> pool_size, Array<PrimExpr> strides,
                   Array<PrimExpr> padding, Array<PrimExpr> dilation, String layout,
                   String out_layout, bool ceil_mode) {
  auto attrs = make_object<MaxPool2DAttrs>();
  attrs->pool_size = pool_size;
  attrs->strides = strides;
  attrs->padding = padding;
  attrs->dilation = dilation;
  attrs->layout = layout;
  attrs->out_layout = out_layout;
  attrs->ceil_mode = ceil_mode;
  static const Op& op = Op::Get("relax.nn.max_pool2d");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.nn.max_pool2d").set_body_typed(MakeMaxPool2D);

/* relax.nn.adaptive_avg_pool2d */
TVM_REGISTER_NODE_TYPE(AdaptivePool2DAttrs);

RELAX_REGISTER_OP("relax.nn.adaptive_avg_pool2d")
    .set_attrs_type<AdaptivePool2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferShape>("FInferShape", InferShapeAdaptiveAvgPool2D)
    .set_attr<FInferType>("FInferType", InferTypeUnaryBroadcast);

Expr MakeAdaptiveAvgPool2D(Expr data, Optional<Array<PrimExpr>> output_size, String layout) {
  ObjectPtr<AdaptivePool2DAttrs> attrs = make_object<AdaptivePool2DAttrs>();
  attrs->output_size = std::move(output_size);
  attrs->layout = std::move(layout);

  const static Op& op = Op::Get("relax.nn.adaptive_avg_pool2d");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.adaptive_avg_pool2d").set_body_typed(MakeAdaptiveAvgPool2D);

Expr InferShapeAdaptiveAvgPool2D(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "AdaptiveAvgPool2d op should have 1 argument");
  }

  const auto* input_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
  if (input_shape == nullptr) {
    return RuntimeDepShape();
  }
  if (input_shape->values.size() != 4) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The resize2d operator expects the input data to be a tensor of 4 "
                          "dimensions. However, the given data has "
                       << input_shape->values.size() << " dimensions");
  }
  if (attrs->layout->size != 4) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The resize2d operator expects the input layout to be a string of length 4, containing "
           "letters \"N\", \"C\", \"H\", \"W\". However, the given layout is "
        << attrs->layout);
  }

  if (!attrs->output_size.defined()) {
    return GetRef<ShapeExpr>(input_shape);
  }
  int batch_axis = -1;
  int height_axis = -1;
  int width_axis = -1;
  int channel_axis = -1;
  for (int i = 0; i < 4; ++i) {
    char letter = attrs->layout.at(i);
    if (letter == 'N') {
      batch_axis = i;
    } else if (letter == 'H') {
      height_axis = i;
    } else if (letter == 'W') {
      width_axis = i;
    } else if (letter == 'C') {
      channel_axis = i;
    }
  }
  if (batch_axis == -1 || height_axis == -1 || width_axis == -1 || channel_axis == -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The adaptive_avg_pool2d operator expects the input layout to be a string of length 4, "
           "containing letters \"N\", \"C\", \"H\", \"W\". However, the given layout is "
        << attrs->layout);
  }

  Array<PrimExpr> output_size = attrs->output_size.value();
  if (output_size.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The adaptive_avg_pool2d operator expects the input size to have exactly "
                          "two elements. However, the given size is "
                       << output_size << ", which contains " << output_size.size() << " elements");
  }

  Array<PrimExpr> output_shape;
  output_shape.resize(4);
  output_shape.Set(batch_axis, input_shape->values[batch_axis]);
  output_shape.Set(height_axis, output_size[0]);
  output_shape.Set(width_axis, output_size[1]);
  output_shape.Set(channel_axis, input_shape->values[channel_axis]);
  return ShapeExpr(output_shape);
}

}  // namespace relax
}  // namespace tvm
