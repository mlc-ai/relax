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

#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>

#include <string>

namespace tvm {
namespace relax {

using relay::Call;

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

/*!
 * \brief Computation description interface.
 *
 * \note This function have a special convention
 *  for functions with tuple input/output.
 *
 *  So far we restrict tuple support to the following case:
 *  - Function which takes a single tuple as input.
 *  - Function which outputs a single tuple.
 *
 *  In both cases, the tuple is flattened as array.
 *
 * \param attrs The attribute of the primitive
 * \param inputs The input tensors.
 * \param out_type The output type information
 &                 these are always placeholders.
 * \return The output compute description of the operator.
 */
using FTVMCompute = runtime::TypedPackedFunc<Array<te::Tensor>(
    const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type)>;

/*! \brief Attributes used in MaxPool2d operator */
struct MaxPool2DAttrs : public tvm::AttrsNode<MaxPool2DAttrs> {
  Array<PrimExpr> pool_size;
  Array<PrimExpr> strides;
  Array<PrimExpr> padding;
  Array<PrimExpr> dilation;
  tvm::String layout;
  tvm::String out_layout;
  bool ceil_mode;

  TVM_DECLARE_ATTRS(MaxPool2DAttrs, "relax.attrs.MaxPool2DAttrs") {
    TVM_ATTR_FIELD(pool_size).describe("Size of the pooling windows.");
    TVM_ATTR_FIELD(strides)
        .set_default(Array<PrimExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<PrimExpr>({1, 1}))
        .describe("Specifies the dilation of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<PrimExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Pooling is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(ceil_mode).set_default(false).describe(
        "When true, will use ceil instead of floor to compute the output shape.");
  }
};  // struct MaxPool2dAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2DAttrs : public tvm::AttrsNode<Conv2DAttrs> {
  Array<PrimExpr> strides;
  Array<PrimExpr> padding;
  Array<PrimExpr> dilation;
  int groups;
  PrimExpr channels;
  Array<PrimExpr> kernel_size;
  String data_layout;
  String kernel_layout;
  String out_layout;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(Conv2DAttrs, "relax.attrs.Conv2DAttrs") {
    TVM_ATTR_FIELD(strides)
        .set_default(Array<PrimExpr>({1, 1}))
        .describe("Specifies the strides of the convolution.");
    TVM_ATTR_FIELD(padding)
        .set_default(Array<PrimExpr>({0, 0}))
        .describe(
            "If padding is non-zero, then the input is implicitly zero-padded"
            "Padding support both symmetric and asymmetric as"
            "one int : same padding used on all sides"
            "two int : bottom, right will use same padding as top, left"
            "four int : padding width in the order of (top, left, bottom, right)");
    TVM_ATTR_FIELD(dilation)
        .set_default(Array<PrimExpr>({1, 1}))
        .describe("Specifies the dilation rate to use for dilated convolution.");
    TVM_ATTR_FIELD(groups).set_default(1).describe(
        "Controls the connections between inputs and outputs."
        "At groups=1, all inputs are convolved to all outputs."
        "At groups=2, the operation becomes equivalent to having two convolution"
        "layers side by side, each seeing half the input channels, and producing"
        "half the output channels, and both subsequently concatenated.");
    TVM_ATTR_FIELD(channels)
        .describe(
            "The number of output channels in the convolution."
            " If it is not set, inferred by shape of the weight.")
        .set_default(NullValue<PrimExpr>());
    TVM_ATTR_FIELD(kernel_size)
        .describe("Specifies the dimensions of the convolution window.")
        .set_default(NullValue<Array<PrimExpr>>());
    TVM_ATTR_FIELD(data_layout)
        .set_default("NCHW")
        .describe(
            "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Convolution is applied on the 'H' and"
            "'W' dimensions.");
    TVM_ATTR_FIELD(kernel_layout)
        .set_default("OIHW")
        .describe(
            "Dimension ordering of weight. Can be 'OIHW', 'OIHW16o16i', etc."
            "'O', 'I', 'H', 'W' stands for num_filter, input_channel, height, and width"
            "dimensions respectively.");
    TVM_ATTR_FIELD(out_layout)
        .set_default("")
        .describe(
            "Dimension ordering of output. Can be 'NCHW', 'NHWC', etc."
            "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
            "dimensions respectively. Default to be same as input layout.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};  // struct Conv2dAttrs

/*! \brief Attributes for dense operator */
struct DenseAttrs : public tvm::AttrsNode<DenseAttrs> {
  PrimExpr units;
  DataType out_dtype;

  TVM_DECLARE_ATTRS(DenseAttrs, "relax.attrs.DenseAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in softmax operators */
struct SoftmaxAttrs : public tvm::AttrsNode<SoftmaxAttrs> {
  int axis;

  TVM_DECLARE_ATTRS(SoftmaxAttrs, "relax.attrs.SoftmaxAttrs") {
    TVM_ATTR_FIELD(axis).set_default(-1).describe("The axis to sum over when computing softmax.");
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

/*! \brief Attributes used in transpose operator */
struct TransposeAttrs : public tvm::AttrsNode<TransposeAttrs> {
  Optional<Array<Integer>> axes;

  TVM_DECLARE_ATTRS(TransposeAttrs, "relax.attrs.TransposeAttrs") {
    TVM_ATTR_FIELD(axes)
        .describe("The target axes order, reverse order if not specified.")
        .set_default(Optional<Array<Integer>>{NullOpt});
  }
};  // struct TransposeAttrs

/*! \brief Attributes used in batch_norm operator */
struct BatchNormAttrs : public tvm::AttrsNode<BatchNormAttrs> {
  int axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(BatchNormAttrs, "relax.attrs.BatchNormAttrs") {
    TVM_ATTR_FIELD(axis).describe("Specify which shape axis denotes the channel.").set_default(1);
    TVM_ATTR_FIELD(epsilon)
        .describe("Small float added to variance to avoid dividing by zero")
        .set_default(1e-5);
    TVM_ATTR_FIELD(center)
        .describe("If True, add offset of beta to normalized tensor. If False, beta is ignored")
        .set_default(true);
    TVM_ATTR_FIELD(scale)
        .describe(
            "If True, multiply by gamma. If False, gamma is not used. "
            "When the next layer is piecewise linear (also, e.g., nn.relu), "
            "this can be disabled since the scaling will be done by the next layer.")
        .set_default(true);
  }
};  // struct BatchNormAttrs

/*! \brief Attributes used in expand_dims operators */
struct ExpandDimsAttrs : public tvm::AttrsNode<ExpandDimsAttrs> {
  Array<Integer> axis;

  TVM_DECLARE_ATTRS(ExpandDimsAttrs, "relax.attrs.ExpandDimsAttrs") {
    TVM_ATTR_FIELD(axis).describe(
        "The axes at which the input array is expanded."
        "Each element should lie in range `[-data.ndim - 1, data.ndim]`."
        "If `axis < 0`, it is the first axis inserted;"
        "If `axis >= 0`, it is the last axis inserted in Python's negative indexing.");
  }
};  // struct ExpandDimsAttrs

/*! \brief Attributes used in squeeze operators */
struct SqueezeAttrs : public tvm::AttrsNode<SqueezeAttrs> {
  Optional<Array<Integer>> axis;

  TVM_DECLARE_ATTRS(SqueezeAttrs, "relax.attrs.SqueezeAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The axis to squeeze in the input tensor."
            "If `axis = None`, all axis of dimension 1 get squeezed;"
            "Else, the dimension in axes get squeezed."
            "It is an error if an axis does not has dimension 1.")
        .set_default(Optional<Array<Integer>>{NullOpt});
  }
};  // struct SqueezeAttrs

/*! \brief Attributes used in concatenate operators */
struct ConcatenateAttrs : public tvm::AttrsNode<ConcatenateAttrs> {
  Optional<Integer> axis;

  TVM_DECLARE_ATTRS(ConcatenateAttrs, "relax.attrs.ConcatenateAttrs") {
    TVM_ATTR_FIELD(axis)
        .describe(
            "The axis at which the input arrays are concatenated."
            "Should lie in range `[-ndim, ndim)`.")
        .set_default(Integer(0));
  }
};  // struct ConcatenateAttrs

/*! \brief Attributes used in dropout operator */
struct DropoutAttrs : public tvm::AttrsNode<DropoutAttrs> {
  double rate;

  TVM_DECLARE_ATTRS(DropoutAttrs, "relax.attrs.DropoutAttrs") {
    TVM_ATTR_FIELD(rate)
        .describe("Fraction of the input that gets dropped out during training time")
        .set_default(0.5);
  }
};  // struct DropoutAttrs

/*! \brief Attributes used in layer_norm operator */
struct LayerNormAttrs : public tvm::AttrsNode<LayerNormAttrs> {
  Array<Integer> axis;
  double epsilon;
  bool center;
  bool scale;

  TVM_DECLARE_ATTRS(LayerNormAttrs, "relax.attrs.LayerNormAttrs") {
    TVM_ATTR_FIELD(axis).set_default(Array<Integer>{Integer(-1)});
    TVM_ATTR_FIELD(epsilon)
        .describe("Small float added to variance to avoid dividing by zero")
        .set_default(1e-5);
    TVM_ATTR_FIELD(center)
        .describe("If True, add offset of beta to normalized tensor. If False, beta is ignored")
        .set_default(true);
    TVM_ATTR_FIELD(scale)
        .describe(
            "If True, multiply by gamma. If False, gamma is not used. "
            "When the next layer is piecewise linear (also, e.g., nn.relu), "
            "this can be disabled since the scaling will be done by the next layer.")
        .set_default(true);
  }
};  // struct LayerNormAttrs

/*! \brief Attributes for reduction operators */
struct ReduceAttrs : public tvm::AttrsNode<ReduceAttrs> {
  Optional<Array<Integer>> axis;
  bool keepdims;

  TVM_DECLARE_ATTRS(ReduceAttrs, "relax.attrs.ReduceAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(Optional<Array<Integer>>{NullOpt})
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");
    TVM_ATTR_FIELD(keepdims).set_default(false).describe(
        "If this is set to `True`, the reduced axes are left "
        "in the result as dimension with size one.");
  }
};  // struct ReduceAttrs

/*! \brief Attributes used in cumsum operator */
struct CumsumAttrs : public tvm::AttrsNode<CumsumAttrs> {
  Optional<Integer> axis;

  TVM_DECLARE_ATTRS(CumsumAttrs, "relax.attrs.CumsumAttrs") {
    TVM_ATTR_FIELD(axis).set_default(Optional<Integer>{NullOpt});
  }
};  // struct CumsumAttrs

/*! \brief Attributes used in trilu operator */
struct TriluAttrs : public tvm::AttrsNode<TriluAttrs> {
  int k;
  bool is_upper;

  TVM_DECLARE_ATTRS(TriluAttrs, "relax.attrs.TriluAttrs") {
    TVM_ATTR_FIELD(k).describe(
        "The number of diagonals above or below the main diagonal to exclude or include.");
    TVM_ATTR_FIELD(is_upper).set_default(true).describe(
        "Whether to keep the upper or lower half of the diagonal.");
  }
};  // struct TriluAttrs

/*! \brief Attributes used in cast operator */
struct CastAttrs : public tvm::AttrsNode<CastAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(CastAttrs, "relax.attrs.CastAttrs") {
    TVM_ATTR_FIELD(dtype).describe("Target data type");
  }
};  // struct CastAttrs.

/*! \brief Attributes used in take operator */
struct TakeAttrs : public tvm::AttrsNode<TakeAttrs> {
  Optional<Integer> axis;
  int batch_dims;
  String mode;

  TVM_DECLARE_ATTRS(TakeAttrs, "relax.attrs.TakeAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(Optional<Integer>{NullOpt})
        .describe("The axis over which to select values.");
    TVM_ATTR_FIELD(batch_dims)
        .set_default(0)
        .describe("The batch_dims over which to select values.");
    TVM_ATTR_FIELD(mode).set_default("clip").describe(
        "Specify how out-of-bound indices will behave."
        "clip - clip to the range (default)"
        "wrap - wrap around the indices"
        "fast - no clip or wrap around (user must make sure indices are in-bound)");
  }
};  // struct TakeAttrs

/*! \brief Attributes used in full operator */
struct FullAttrs : public tvm::AttrsNode<FullAttrs> {
  DataType dtype;

  TVM_DECLARE_ATTRS(FullAttrs, "relax.attrs.FullAttrs") {
    TVM_ATTR_FIELD(dtype).describe("Target data type.");
  }
};  // struct FullAttrs

/*! \brief Attributes used in split operator */
struct SplitAttrs : public tvm::AttrsNode<SplitAttrs> {
  ObjectRef indices_or_sections;
  int axis;

  TVM_DECLARE_ATTRS(SplitAttrs, "relax.attrs.SplitAttrs") {
    TVM_ATTR_FIELD(indices_or_sections)
        .describe("The input array of indices or the number of split sections.");
    TVM_ATTR_FIELD(axis).describe("The axis to be splitted");
  }
};  // struct SplitAttrs

/*! \brief Attributes used in strided_slice operator */
struct StridedSliceAttrs : public tvm::AttrsNode<StridedSliceAttrs> {
  Array<PrimExpr> begin;
  Array<PrimExpr> end;
  Optional<Array<PrimExpr>> strides;
  Optional<Array<Integer>> axes;
  String slice_mode;

  TVM_DECLARE_ATTRS(StridedSliceAttrs, "relax.attrs.StridedSliceAttrs") {
    TVM_ATTR_FIELD(begin).describe("Indices for begin of slice, begin index is also inclusive");
    TVM_ATTR_FIELD(end).describe("Indices for end of slice, end index is exclusive");
    TVM_ATTR_FIELD(strides).describe(
        "Stride values of the slice, a stride can be negative, which causes a reverse slice.");
    TVM_ATTR_FIELD(axes).describe(
        "Axes along which slicing is applied. When it is specified, the length of begin, end, "
        "strides, and axes must be equal.");
    TVM_ATTR_FIELD(slice_mode)
        .set_default("end")
        .describe(
            "The slice mode [end, size]."
            "end - The default slice mode, ending indices for the slice."
            "size - The input strides will be ignored, input end in this mode indicates the size"
            "of a slice starting at the location specified by begin. If end[i] is -1,"
            "all remaining elements in that dimension are included in the slice");
  }
};  // struct StridedSliceAttrs

/*! \brief Attributes used in image resize2d operator */
struct Resize2DAttrs : public tvm::AttrsNode<Resize2DAttrs> {
  Array<PrimExpr> size;
  Array<FloatImm> roi;
  String layout;
  String method;
  String coordinate_transformation_mode;
  String rounding_method;
  double cubic_alpha;
  int cubic_exclude;
  double extrapolation_value;

  TVM_DECLARE_ATTRS(Resize2DAttrs, "relax.attrs.Resize2DAttrs") {
    TVM_ATTR_FIELD(size).describe("Output image size.");
    TVM_ATTR_FIELD(roi).describe(
        "Region of Interest for coordinate transformation mode 'tf_crop_and_resize'");
    TVM_ATTR_FIELD(layout).set_default("NCHW").describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Resize is applied on the 'H' and"
        "'W' dimensions.");
    TVM_ATTR_FIELD(method).set_default("linear").describe(
        "Specify the mode to use for scaling."
        "nearest_neighbor -  Nearest Neighbor"
        "linear - Bilinear Interpolation"
        "cubic - Bicubic Interpolation");
    TVM_ATTR_FIELD(coordinate_transformation_mode)
        .set_default("half_pixel")
        .describe(
            "Describes how to transform the coordinate in the resized tensor"
            "to the coordinate in the original tensor."
            "Refer to the ONNX Resize operator specification for details"
            "Available options are half_pixel, align_corners and asymmetric");
    TVM_ATTR_FIELD(rounding_method)
        .set_default("round")
        .describe(
            "indicates how to find the \"nearest\" pixel in nearest_neighbor method"
            "Available options are round, floor, and ceil.");
    TVM_ATTR_FIELD(cubic_alpha)
        .set_default(-0.5)
        .describe("Spline Coefficient for Bicubic Interpolation");
    TVM_ATTR_FIELD(cubic_exclude)
        .set_default(0)
        .describe("Flag to exclude exterior of the image during bicubic interpolation");
    TVM_ATTR_FIELD(extrapolation_value)
        .set_default(0.0)
        .describe("Value to return when roi is outside of the image");
  }
};  // struct Resize2dAttrs

/*! \brief Attributes for 2d adaptive pool operator */
struct AdaptivePool2DAttrs : public tvm::AttrsNode<AdaptivePool2DAttrs> {
  Optional<Array<PrimExpr>> output_size;
  String layout;

  TVM_DECLARE_ATTRS(AdaptivePool2DAttrs, "relax.attrs.AdaptivePool2DAttrs") {
    TVM_ATTR_FIELD(output_size).describe("Output height and width.");
    TVM_ATTR_FIELD(layout).describe(
        "Dimension ordering of input data. Can be 'NCHW', 'NHWC', etc."
        "'N', 'C', 'H', 'W' stands for batch, channel, height, and width"
        "dimensions respectively. Pooling is applied on the 'H' and"
        "'W' dimensions.");
  }
};  // struct AdaptivePool2DAttrs

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_ATTR_TYPES_H_
