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
 * \file manipulate.cc
 * \brief Manipulation operators.
 */

#include <numeric>

#include "../op_common.h"

namespace tvm {
namespace relax {

PrimExpr ComputeShapeProduct(const Array<PrimExpr>& shape_values) {
  PrimExpr shape_prod = IntImm(DataType::Int(64), 1);
  for (PrimExpr value : shape_values) {
    shape_prod *= value;
  }
  return shape_prod;
}

/* relax.reshape */
Expr ConvertNewShapeToExpr(const Expr& data, const ObjectRef& shape) {
  if (const auto* e = shape.as<ExprNode>()) {
    return GetRef<Expr>(e);
  }

  const auto* array = shape.as<ArrayNode>();
  CHECK(array != nullptr) << "Reshape only expects the input new shape to be either an Expr or an "
                             "Array of PrimExprs. However, the given new shape is "
                          << shape;
  int dim_to_infer = -1;
  PrimExpr new_shape_prod = IntImm(DataType::Int(64), 1);
  for (int i = 0; i < static_cast<int>(array->size()); ++i) {
    const auto* _len = array->at(i).as<PrimExprNode>();
    CHECK(_len != nullptr) << "Reshape only expects the input new shape to be either an Expr or an "
                              "Array of PrimExprs. However, the given new shape is "
                           << shape;
    PrimExpr len = GetRef<PrimExpr>(_len);
    CHECK(len->dtype.is_int()) << "Reshape requires the new shape values to be all "
                                  "integers. However, the give new shape is "
                               << shape;
    const auto* int_len = len.as<IntImmNode>();
    if (int_len != nullptr && int_len->value == -1) {
      CHECK_EQ(dim_to_infer, -1) << "Reshape accepts at most one \"-1\" in the new shape. However, "
                                    "there are multiple \"-1\" in the given new shape  "
                                 << shape;
      dim_to_infer = i;
    } else {
      CHECK(int_len == nullptr || int_len->value > 0)
          << "Reshape requires all values in the new shape to be positive except a single \"-1\". "
             "However, the given new shape is "
          << shape;
      // We expect any symbolic not to signal the intent of -1, and therefore do no check for
      // symbolic value here.
      new_shape_prod = new_shape_prod * len;
    }
  }

  Array<PrimExpr> array_ref = GetRef<Array<PrimExpr>>(array);
  // When there is no dimension to infer, just return the input array as ShapeExpr.
  if (dim_to_infer == -1) {
    return ShapeExpr(array_ref);
  }

  // Otherwise, we require the input tensor to have known shape value for inference.
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(data);
  CHECK(data_sinfo != nullptr)
      << "Reshape expects the input data to be a Tensor. However, the given input is "
      << data->struct_info_->GetTypeKey();
  CHECK(data_sinfo->shape.defined())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input has no shape.";
  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value());
  CHECK(shape_sinfo != nullptr && shape_sinfo->values.defined())
      << "Reshape expects the input tensor to have known shape when there is some dimension length "
         "to infer. However, the given input shape is "
      << data_sinfo->shape << " whose shape value is unknown.";

  arith::Analyzer analyzer;
  PrimExpr old_shape_prod = ComputeShapeProduct(shape_sinfo->values.value());
  array_ref.Set(dim_to_infer, analyzer.Simplify(floordiv(old_shape_prod, new_shape_prod)));
  return ShapeExpr(array_ref);
}

Expr MakeReshape(Expr data, ObjectRef shape) {
  Expr shape_in_expr = ConvertNewShapeToExpr(data, shape);
  static const Op& op = Op::Get("relax.reshape");
  return Call(op, {std::move(data), std::move(shape_in_expr)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.reshape").set_body_typed(MakeReshape);

StructInfo InferStructInfoReshape(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Reshape op should take 2 arguments");
  }
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* new_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Reshape requires the input data to be Tensor. However, the given one is "
                     << call->args[0]->struct_info_->GetTypeKey());
  }
  if (new_shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Reshape requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->struct_info_->GetTypeKey());
  }

  Optional<Array<PrimExpr>> old_shape_values;
  if (data_sinfo->shape.defined()) {
    const auto* old_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value());
    ICHECK_NOTNULL(old_shape_sinfo);
    old_shape_values = old_shape_sinfo->values;
  }

  if (new_shape_sinfo->values.defined() && old_shape_values.defined()) {
    PrimExpr new_shape_prod = ComputeShapeProduct(new_shape_sinfo->values.value());
    PrimExpr old_shape_prod = ComputeShapeProduct(old_shape_values.value());
    if (ctx->GetAnalyzer()->CanProve(old_shape_prod != new_shape_prod)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Reshape expects the new shape to be convertible from the old shape. "
                          "However, the old shape is "
                       << data_sinfo->shape << ", with product " << old_shape_prod
                       << ", while the new shape is " << call->args[1] << ", with product "
                       << new_shape_prod);
    }
  }
  return TensorStructInfo(call->args[1], data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.reshape")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("shape", "Shape", "The input new shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReshape);

/* relax.permute_dims */
TVM_REGISTER_NODE_TYPE(PermuteDimsAttrs);

Expr MakePermuteDims(Expr data, Optional<Array<Integer>> axes) {
  ObjectPtr<PermuteDimsAttrs> attrs = make_object<PermuteDimsAttrs>();
  attrs->axes = std::move(axes);

  static const Op& op = Op::Get("relax.permute_dims");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.permute_dims").set_body_typed(MakePermuteDims);

bool IsIdentityPermutation(const std::vector<int>& permutation) {
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

StructInfo InferStructInfoPermuteDims(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  if (data_sinfo->IsUnknownNdim()) {
    if (attrs->axes.defined()) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "PermuteDims cannot be performed when the input tensor " << data_sinfo
                       << " ndim is unknown while the given number of axes " << attrs->axes.value()
                       << " is clear. Please use MatchCast to match the input tensor to a specific "
                          "ndim before doing PermuteDims.");
    }
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  if (attrs->axes.defined()) {
    int n_axis = attrs->axes.value().size();
    if (n_axis != data_sinfo->ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "PermuteDims expects the number of input axes to equal the ndim of the "
                          "input tensor. However, the tensor ndim is "
                       << data_sinfo->ndim << " while the given number of axes is " << n_axis);
    }
  }

  std::vector<int> axes;
  if (attrs->axes.defined()) {
    axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axes.value());
  } else {
    // Construct the reverse permutation via std::iota
    axes.resize(data_sinfo->ndim);
    std::iota(axes.rbegin(), axes.rend(), 0);
  }
  if (IsIdentityPermutation(axes)) {
    return data_sinfo;
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim);
  }
  std::vector<PrimExpr> new_shape;
  new_shape.reserve(data_sinfo->ndim);
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    new_shape.push_back(data_shape->values[axes[i]]);
  }
  return TensorStructInfo(ShapeExpr(new_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.permute_dims")
    .set_attrs_type<PermuteDimsAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "nD Tensor", "input tensor to be permute_dimsd")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoPermuteDims);

/* relax.expand_dims */
TVM_REGISTER_NODE_TYPE(ExpandDimsAttrs);

Expr MakeExpandDims(Expr data, Array<Integer> axis) {
  ObjectPtr<ExpandDimsAttrs> attrs = make_object<ExpandDimsAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.expand_dims");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.expand_dims").set_body_typed(MakeExpandDims);

StructInfo InferStructInfoExpandDims(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  if (attrs->axis.empty()) {
    return data_sinfo;
  }

  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  int n_new_dim = attrs->axis.size();
  int output_ndim = data_sinfo->ndim + n_new_dim;
  std::vector<int> axes = NormalizeAxes(call, ctx, output_ndim, attrs->axis);

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, output_ndim);
  }

  std::vector<PrimExpr> output_shape;
  output_shape.resize(output_ndim, PrimExpr());
  for (int i = 0; i < n_new_dim; ++i) {
    output_shape[axes[i]] = IntImm(DataType::Int(64), 1);
  }

  int i_data_shape = 0;
  for (int i = 0; i < output_ndim; ++i) {
    if (output_shape[i].defined()) {
      continue;
    }
    ICHECK_LT(i_data_shape, data_sinfo->ndim);
    output_shape[i] = data_shape->values[i_data_shape];
    ++i_data_shape;
  }
  ICHECK_EQ(i_data_shape, data_sinfo->ndim);
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.expand_dims")
    .set_num_inputs(1)
    .set_attrs_type<ExpandDimsAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoExpandDims);

/* relax.squeeze */
TVM_REGISTER_NODE_TYPE(SqueezeAttrs);

Expr MakeSqueeze(Expr data, Optional<Array<Integer>> axis) {
  ObjectPtr<SqueezeAttrs> attrs = make_object<SqueezeAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.squeeze");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.squeeze").set_body_typed(MakeSqueeze);

StructInfo InferStructInfoSqueeze(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  if (attrs->axis.defined() && attrs->axis.value().empty()) {
    return data_sinfo;
  }

  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  Optional<Array<PrimExpr>> shape_value;
  if (data_sinfo->shape.defined()) {
    shape_value = Downcast<ShapeStructInfo>(data_sinfo->shape.value()->struct_info_)->values;
  }

  std::vector<bool> axis_removal_mask;
  axis_removal_mask.resize(data_sinfo->ndim, /*value=*/false);

  if (attrs->axis.defined()) {
    std::vector<int> axes = NormalizeAxes(call, ctx, data_sinfo->ndim, attrs->axis.value());

    if (!shape_value.defined()) {
      return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim - axes.size());
    }
    for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
      // When `axis` is given, the dim lengths at the axes must be static constant integer 1.
      const auto* int_len = shape_value.value()[axes[i]].as<IntImmNode>();
      if (int_len == nullptr || int_len->value != 1) {
        // We would like to ensure safety, and therefore placed a stronger requirement for user to
        // use MatchCast.
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Squeeze expects the input tensor shape values at the given axis "
                            "positions to be all 1. However, the tensor shape at axis "
                         << axes[i] << " is " << shape_value.value()[axes[i]]
                         << " which is not 1. If it is symbolic, please use MatchCast to cast it "
                            "to 1 before doing Squeeze.");
      }
      axis_removal_mask[axes[i]] = true;
    }
  } else {
    // When `axis` is not defined, squeeze all unit-length dimensions.
    // Note: This is a less well-defined path in Array API standard's squeeze
    // (https://data-apis.org/array-api/latest/API_specification/generated/array_api.squeeze.html).
    // Consider discourage usage later.
    if (!shape_value.defined()) {
      return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
    }
    for (int i = 0; i < data_sinfo->ndim; ++i) {
      // Whenever a dimension length is symbolic, fall back to unknown ndim.
      const auto* int_len = shape_value.value()[i].as<IntImmNode>();
      if (int_len == nullptr) {
        return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
      }
      if (int_len->value == 1) {
        axis_removal_mask[i] = true;
      }
    }
  }

  std::vector<PrimExpr> output_shape;
  output_shape.reserve(data_sinfo->ndim - axis_removal_mask.size());
  for (int i = 0; i < data_sinfo->ndim; ++i) {
    if (!axis_removal_mask[i]) {
      output_shape.push_back(shape_value.value()[i]);
    }
  }

  if (data_sinfo->shape.value()->IsInstance<VarNode>()) {
    if (static_cast<int>(output_shape.size()) == data_sinfo->ndim) {
      return data_sinfo;
    } else if (attrs->axis.defined()) {
      return TensorStructInfo(data_sinfo->dtype, output_shape.size());
    } else {
      return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
    }
  } else {
    return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype);
  }
}

TVM_REGISTER_OP("relax.squeeze")
    .set_num_inputs(1)
    .set_attrs_type<SqueezeAttrs>()
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSqueeze);

/* relax.flatten */
Expr MakeFlatten(Expr data) {
  static const Op& op = Op::Get("relax.flatten");
  return Call(op, {std::move(data)}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.flatten").set_body_typed(MakeFlatten);

StructInfo InferStructInfoFlatten(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  if (data_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/1);
  } else if (data_sinfo->ndim == 0) {
    return TensorStructInfo(ShapeExpr({1}), data_sinfo->dtype);
  } else if (data_sinfo->ndim == 1) {
    return data_sinfo;
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/1);
  }
  PrimExpr shape_prod = ComputeShapeProduct(data_shape->values);
  return TensorStructInfo(ShapeExpr({std::move(shape_prod)}), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.flatten")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFlatten);

}  // namespace relax
}  // namespace tvm
