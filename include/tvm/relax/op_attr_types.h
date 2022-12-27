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
 * \file tvm/relax/op_attr_types.h
 * \brief Data structures that can appear in operator attributes.
 */
#ifndef TVM_RELAX_OP_ATTR_TYPES_H_
#define TVM_RELAX_OP_ATTR_TYPES_H_

#include <tvm/relax/struct_info.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>

#include <string>

namespace tvm {
namespace relax {

/*!
 * \brief Infer the output shape for operators. This function will
 * be invoked to fill the \p shape_ field of expressions.
 * \param call The call node.
 * \param diag_ctx The diagnostic context for reporting errors.
 * \return The inferred output shape expression.
 */
using FInferShape =
    runtime::TypedPackedFunc<RelayExpr(const Call& call, DiagnosticContext diag_ctx)>;

/*!
 * \brief Infer output struct info given the call
 *
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 */
using FInferStructInfo =
    runtime::TypedPackedFunc<StructInfo(const Call& call, const BlockBuilder& ctx)>;

/*!
 * \brief Infer the output type for operators. This function will
 * be invoked to fill the \p checked_type_ field of expressions.
 * \param call The call node.
 * \param diag_ctx The diagnostic context for reporting errors.
 * \return The inferred output type.
 */
using FInferType = runtime::TypedPackedFunc<Type(const Call& call, DiagnosticContext diag_ctx)>;

/*!
 * \brief Packed function implementation for operators. The relax operator will be lowered to
 * this packed function call during codegen.
 */
using FCallPacked = String;

/*! \brief Attributes used in Conv2d operator */
struct Conv2DAttrs : public tvm::AttrsNode<Conv2DAttrs> {
  Array<PrimExpr> strides;
  Array<PrimExpr> padding;
  Array<PrimExpr> dilation;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DAttrs, "relax.attrs.Conv2DAttrs") {
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "two int : bottom, right will use same padding as top, left"
        "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(dilation).describe(
        "Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(data_layout)
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");
    TVM_ATTR_FIELD(out_dtype).describe(
        "Output data type, set to explicit type under mixed precision setting");
  }
};  // struct Conv2dAttrs

/*! \brief Attributes used in max_pool2d operator */
struct MaxPool2DAttrs : public tvm::AttrsNode<MaxPool2DAttrs> {
  Array<PrimExpr> pool_size;
  Array<PrimExpr> strides;
  Array<PrimExpr> padding;
  Array<PrimExpr> dilation;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(MaxPool2DAttrs, "relax.attrs.MaxPool2DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides).describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation).describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding).describe(
        "If padding is non-zero, then the input is implicitly zero-padded"
        "Padding support both symmetric and asymmetric as"
        "one int : same padding used on all sides"
        "two int : bottom, right will use same padding as top, left"
        "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
  }
};  // struct MaxPool2dAttrs

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrs : public tvm::AttrsNode<AdaptivePool2DAttrs> {
  Optional<Array<PrimExpr>> output_size;
  String layout;
  String out_layout;

  TVM_DECLARE_ATTRS(AdaptivePool2DAttrs, "relax.attrs.AdaptivePool2DAttrs") {
    TVM_ATTR_FIELD(output_size).describe("Output height and width.");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
  }
};  // struct AdaptivePool2DAttrs

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public tvm::AttrsNode<SoftmaxAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SoftmaxAttrs, "relax.attrs.SoftmaxAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis to sum over when computing softmax.");
  }
};

/*! \brief Attributes used in unique operator */
struct UniqueAttrs : public tvm::AttrsNode<UniqueAttrs> {
  bool sorted;
  bool return_inverse;
  bool return_counts;
  int dim;
  TVM_DECLARE_ATTRS(UniqueAttrs, "relax.attrs.UniqueAttrs") {
    TVM_ATTR_FIELD(sorted)
        .describe(
            "Whether to sort the unique elements in ascending order before returning as output.")
        .set_default(true);
    TVM_ATTR_FIELD(return_inverse)
        .describe(
            "Whether to return an additional tensor with indices for where elements in the "
            "original input ended up in the returned unique list.")
        .set_default(false);
    TVM_ATTR_FIELD(return_counts)
        .describe("Whether to return an additional tensor with counts of each unique elements")
        .set_default(false);
    TVM_ATTR_FIELD(dim)
        .describe(
            "The dimension to apply unique. If negative, the unique of the flattened input is "
            "returned.")
        .set_default(-1);
  }
};  // struct UniqueAttrs

struct PrintAttrs : public tvm::AttrsNode<PrintAttrs> {
  std::string format;
  TVM_DECLARE_ATTRS(PrintAttrs, "relax.attrs.PrintAttrs") {
    TVM_ATTR_FIELD(format)
        .describe("Python-style format string to use for displaying the input. Ignored if empty.")
        .set_default("");
  }
};

struct AssertOpAttrs : public tvm::AttrsNode<AssertOpAttrs> {
  std::string format;
  TVM_DECLARE_ATTRS(AssertOpAttrs, "relax.attrs.AssertOpAttrs") {
    TVM_ATTR_FIELD(format)
        .describe(
            "Python-style format string to use for displaying "
            "an error message if the assert fails. "
            "Ignored if empty.")
        .set_default("");
  }
};

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public tvm::AttrsNode<BatchNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(BatchNormAttrs, "relax.attrs.BatchNormAttrs") {
    TVM_ATTR_FIELD(axis).describe("The axis along which the normalization is applied.");
    TVM_ATTR_FIELD(epsilon).describe("Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).describe(
        "Indicating if the beta offset will be added to the normalized tensor.");
    TVM_ATTR_FIELD(scale).describe("Indicating if the gamma scale will be multiplied.");
  }
};  // struct BatchNormAttrs

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public tvm::AttrsNode<LayerNormAttrs> {
  Array<Integer> axes;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(LayerNormAttrs, "relax.attrs.LayerNormAttrs") {
    TVM_ATTR_FIELD(axes).describe("The axes that along which the normalization is applied.");
    TVM_ATTR_FIELD(epsilon).describe("Small float added to variance to avoid dividing by zero");
    TVM_ATTR_FIELD(center).describe(
        "Indicating if the beta offset will be added to the normalized tensor.");
    TVM_ATTR_FIELD(scale).describe("Indicating if the gamma scale will be multiplied.");
  }
};  // struct LayerNormAttrs

/*! \brief Attributes for matmul operator */
struct MatmulAttrs : public tvm::AttrsNode<MatmulAttrs> {
  DataType out_dtype;

  TVM_DECLARE_ATTRS(MatmulAttrs, "relax.attrs.MatmulAttrs") {
    TVM_ATTR_FIELD(out_dtype).describe("The data type of the output tensor");
  }
};  // struct MatmulAttrs

/*! \brief Attributes used in dropout operator */
struct DropoutAttrs : public tvm::AttrsNode<DropoutAttrs> {
  double rate;

  TVM_DECLARE_ATTRS(DropoutAttrs, "relax.attrs.DropoutAttrs") {
    TVM_ATTR_FIELD(rate).describe(
        "Fraction of the input that gets dropped out during training time");
  }
};  // struct DropoutAttrs

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_ATTR_TYPES_H_
