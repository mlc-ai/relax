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

#include <vector>

namespace tvm {
namespace relax {

/* relax.nn.conv2d */
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

Expr conv2d(Expr data, Expr weight, Array<IntImm> strides, Array<IntImm> padding,
            Array<IntImm> dilation, int groups, String data_layout, String kernel_layout,
            Optional<String> out_layout, DataType out_dtype) {
  padding = GetCompletePadding2D(std::move(padding));
  if (strides.size() == 1) {
    strides.push_back(strides[0]);
  }
  if (dilation.size() == 1) {
    dilation.push_back(dilation[0]);
  }

  CHECK_GT(groups, 0) << "The number of groups in convolution is expected to be positive. However, "
                         "the given number of groups is "
                      << groups;
  CHECK_EQ(strides.size(), 2)
      << "The input strides length is expected to be 2. However, the given strides is " << strides;
  CHECK_EQ(dilation.size(), 2)
      << "The input dilation length is expected to be 2. However, the given dilation is "
      << dilation;
  return MakeConv<Conv2DAttrs>(std::move(data), std::move(weight), std::move(strides),
                               std::move(padding), std::move(dilation), groups, data_layout,
                               std::move(kernel_layout), out_layout.value_or(data_layout),
                               out_dtype, /*op_name=*/"relax.nn.conv2d");
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv2d").set_body_typed(conv2d);

StructInfo InferStructInfoConv2d(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo weight_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->data_layout,  //
                                                    /*tgt_layout=*/"NCHW",          //
                                                    /*tensor_name=*/"data");
  auto [weight_layout, weight2OIHW] = CheckTensorLayout(call, ctx, attrs->kernel_layout,  //
                                                        /*tgt_layout=*/"OIHW",            //
                                                        /*tensor_name=*/"kernel");
  auto [out_layout, out2NCHW] = CheckTensorLayout(call, ctx, attrs->out_layout,  //
                                                  /*tgt_layout=*/"NCHW",         //
                                                  /*tensor_name=*/"output");

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  Optional<ShapeExpr> weight_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, weight_sinfo, weight_layout);

  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, data_sinfo, weight_sinfo)
                           : attrs->out_dtype;
  if (!data_shape.defined() || !weight_shape.defined()) {
    return TensorStructInfo(out_dtype, out_layout.ndim());
  }

  Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> weight_OIHW_shape = weight2OIHW.ForwardShape(weight_shape.value()->values);

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr input_channel_data = data_NCHW_shape[1];
  PrimExpr input_channel_kernel = weight_OIHW_shape[1];
  if (analyzer->CanProve(input_channel_data != input_channel_kernel * attrs->groups)) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "The channel size of the data should equal to the product of input channel size of the "
           "weight and the number of groups. However, the data channel size is "
        << input_channel_data << " while the weight input channel size and number of groups are "
        << input_channel_kernel << " and " << attrs->groups);
  } else if (!analyzer->CanProveEqual(input_channel_data, input_channel_kernel * attrs->groups)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }
  if (analyzer->CanProve(floormod(weight_OIHW_shape[0], attrs->groups) != 0)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Conv2d expects the number of output channels to be divisible by the "
                        "number of groups. However, the number of output channels is "
                     << weight_OIHW_shape[0] << " while the number of groups is " << attrs->groups);
  } else if (!analyzer->CanProveEqual(floormod(weight_OIHW_shape[0], attrs->groups), 0)) {
    // Todo(relax-team): Trust the input shape at this moment, and revisit
    // this condition with runtime shape check
  }

  PrimExpr input_h = data_NCHW_shape[2];
  PrimExpr input_w = data_NCHW_shape[3];
  PrimExpr kernel_h = weight_OIHW_shape[2];
  PrimExpr kernel_w = weight_OIHW_shape[3];
  PrimExpr padding_h = attrs->padding[0] + attrs->padding[2];
  PrimExpr padding_w = attrs->padding[1] + attrs->padding[3];

  std::vector<PrimExpr> out_NCHW_shape;
  out_NCHW_shape.resize(4);
  out_NCHW_shape[0] = data_NCHW_shape[0];
  out_NCHW_shape[1] = weight_OIHW_shape[0];

  PrimExpr numerator_h = input_h + padding_h - attrs->dilation[0] * (kernel_h - 1) - 1;
  PrimExpr numerator_w = input_w + padding_w - attrs->dilation[1] * (kernel_w - 1) - 1;
  out_NCHW_shape[2] = analyzer->Simplify(floordiv(numerator_h, attrs->strides[0]) + 1);
  out_NCHW_shape[3] = analyzer->Simplify(floordiv(numerator_w, attrs->strides[1]) + 1);

  Array<PrimExpr> out_shape = out2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype);
}

InferLayoutOutput InferLayoutConv2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    const VarLayoutMap& var_layout_map) {
  const auto& it = desired_layouts.find("relax.nn.conv2d");
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  LayoutDecision data_layout, weight_layout, output_layout;
  ObjectPtr<Conv2DAttrs> new_attrs = make_object<Conv2DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv2d.
    Layout desired_data_layout = (*it).second[0];
    Layout desired_weight_layout = (*it).second[1];
    Layout desired_output_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
    ICHECK_EQ(desired_data_layout.ndim(), desired_data_layout.ndim_primal()) << "Axis swap only";
    ICHECK_EQ(desired_weight_layout.ndim(), desired_weight_layout.ndim_primal())
        << "Axis swap only";
    ICHECK_EQ(desired_output_layout.ndim(), desired_output_layout.ndim_primal())
        << "Axis swap only";
    data_layout = TransposeLike(InitialLayout(4), attrs->data_layout, desired_data_layout);
    weight_layout = TransposeLike(InitialLayout(4), attrs->kernel_layout, desired_weight_layout);
    output_layout = TransposeLike(InitialLayout(4), attrs->out_layout, desired_output_layout);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    // We don't have a desired layout for conv2d.
    // We can just propagate the layout from the input.
    data_layout = GetLayoutDecision(var_layout_map, call->args[0]);
    weight_layout = GetLayoutDecision(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(attrs->data_layout, InitialLayout(4), data_layout->layout).name();
    new_attrs->kernel_layout =
        TransposeLike(attrs->kernel_layout, InitialLayout(4), weight_layout->layout).name();
    new_attrs->out_layout =
        TransposeLike(attrs->out_layout, InitialLayout(4), output_layout->layout).name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

Call InferMixedPrecisionConv2d(const Call& call, const DataType& out_dtype) {
  const auto* conv2d_attrs = call->attrs.as<Conv2DAttrs>();
  return Downcast<Call>(conv2d(call->args[0], call->args[1], conv2d_attrs->strides,
                               conv2d_attrs->padding, conv2d_attrs->dilation, conv2d_attrs->groups,
                               conv2d_attrs->data_layout, conv2d_attrs->kernel_layout,
                               conv2d_attrs->out_layout, out_dtype));
}

TVM_REGISTER_OP("relax.nn.conv2d")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoConv2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionConv2d)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutConv2d);

}  // namespace relax
}  // namespace tvm
