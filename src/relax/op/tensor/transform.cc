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
 * \file transform.cc
 * \brief Transform operators.
 */

#include <numeric>

#include "../op_common.h"

namespace tvm {
namespace relax {

/* relax.cast */
TVM_REGISTER_NODE_TYPE(CastAttrs);

Expr MakeCast(Expr data, DataType dtype) {
  ObjectPtr<CastAttrs> attrs = make_object<CastAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.cast");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.cast").set_body_typed(MakeCast);

StructInfo InferStructInfoCast(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<CastAttrs>();
  ObjectPtr<TensorStructInfoNode> new_sinfo = make_object<TensorStructInfoNode>(*sinfo.get());
  new_sinfo->dtype = attrs->dtype;
  return TensorStructInfo(new_sinfo);
}

TVM_REGISTER_OP("relax.cast")
    .set_attrs_type<CastAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCast);

/* relax.wrap_param */
TVM_REGISTER_NODE_TYPE(WrapParamAttrs);

Expr MakeWrapParam(Expr data, DataType dtype) {
  ObjectPtr<WrapParamAttrs> attrs = make_object<WrapParamAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.wrap_param");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.wrap_param").set_body_typed(MakeWrapParam);

StructInfo InferStructInfoWrapParam(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* attrs = call->attrs.as<WrapParamAttrs>();
  ObjectPtr<TensorStructInfoNode> new_sinfo = make_object<TensorStructInfoNode>(*sinfo.get());
  new_sinfo->dtype = attrs->dtype;
  return TensorStructInfo(new_sinfo);
}

TVM_REGISTER_OP("relax.wrap_param")
    .set_attrs_type<WrapParamAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoWrapParam);

/* relax.reshape */
TVM_REGISTER_NODE_TYPE(ReshapeAttrs);

Expr MakeReshape(Expr data, ObjectRef new_shape) {
  static const Op& op = Op::Get("relax.reshape");
  if (const auto* expr = new_shape.as<ExprNode>()) {
    return Call(op, {std::move(data), GetRef<Expr>(expr)}, Attrs(), {});
  } else if (const auto* _array = new_shape.as<ArrayNode>()) {
    ObjectPtr<ReshapeAttrs> attrs = make_object<ReshapeAttrs>();
    attrs->new_shape = GetRef<Array<PrimExpr>>(_array);
    return Call(op, {std::move(data)}, Attrs(attrs), {});
  }
  LOG(FATAL) << "Reshape only expects the input new shape to be either an Expr or an Array of "
                "PrimExpr. However, the given one is "
             << new_shape->GetTypeKey();
  throw;
}

TVM_REGISTER_GLOBAL("relax.op.reshape").set_body_typed(MakeReshape);

std::pair<PrimExpr, int> CheckNewShape(const Call& call, const BlockBuilder& ctx,
                                       const Array<PrimExpr>& new_shape, bool allow_dim_inference) {
  int dim_to_infer = -1;
  int new_ndim = new_shape.size();
  PrimExpr new_shape_prod = IntImm(DataType::Int(64), 1);
  for (int i = 0; i < new_ndim; ++i) {
    if (!new_shape[i]->dtype.is_int()) {
      ctx->ReportFatal(Diagnostic::Error(call) << "Reshape requires the new shape values to be all "
                                                  "integers. However, the give new shape is "
                                               << new_shape);
    }
    const auto* int_len = new_shape[i].as<IntImmNode>();
    if (int_len != nullptr && int_len->value == -1) {
      if (!allow_dim_inference) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Reshape expects the new shape not to contain \"-1\" when the data "
                            "shape is not known symbolically. However, the new shape is "
                         << call->args[1]);
      } else if (dim_to_infer != -1) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Reshape accepts at most one \"-1\" in the new shape. However, there "
                            "are multiple \"-1\" in the given one "
                         << call->args[1]);
      }
      dim_to_infer = i;
    } else if (ctx->GetAnalyzer()->CanProveLess(new_shape[i], 1)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Reshape requires all values in the new shape to be positive except a "
                          "single \"-1\". However, the given new shape is "
                       << call->args[1]);
    } else {
      new_shape_prod = new_shape_prod * new_shape[i];
    }
  }
  return {new_shape_prod, dim_to_infer};
}

StructInfo InferStructInfoReshape(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() > 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Reshape op should no more than 2 arguments");
  }
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Reshape requires the input data to be Tensor. However, the given one is "
                     << call->args[0]->struct_info_->GetTypeKey());
  }

  Optional<PrimExpr> old_shape_prod;
  if (data_sinfo->shape.defined()) {
    const auto* old_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(data_sinfo->shape.value());
    ICHECK_NOTNULL(old_shape_sinfo);
    if (old_shape_sinfo->values.defined()) {
      PrimExpr _old_shape_prod = IntImm(DataType::Int(64), 1);
      for (int i = 0; i < data_sinfo->ndim; ++i) {
        _old_shape_prod = _old_shape_prod * old_shape_sinfo->values.value()[i];
      }
      old_shape_prod = _old_shape_prod;
    }
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  if (call->args.size() == 2) {
    ICHECK(call->attrs.get() == nullptr);
    const auto* new_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
    if (new_shape_sinfo == nullptr) {
      ctx->ReportFatal(
          Diagnostic::Error(call)
          << "Reshape requires the input new shape to be Shape. However, the given one is "
          << call->args[1]->struct_info_->GetTypeKey());
    }

    if (new_shape_sinfo->values.defined() && old_shape_prod.defined()) {
      PrimExpr new_shape_prod = IntImm(DataType::Int(64), 1);
      for (int i = 0; i < new_shape_sinfo->ndim; ++i) {
        new_shape_prod = new_shape_prod * new_shape_sinfo->values.value()[i];
      }
      if (analyzer->CanProve(old_shape_prod.value() != new_shape_prod)) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << "Reshape expects the new shape to be convertible from the old shape. "
                            "However, the old shape is "
                         << data_sinfo->shape << ", with product " << old_shape_prod.value()
                         << ", while the new shape is " << call->args[1] << ", with product "
                         << new_shape_prod);
      }
    }
    return TensorStructInfo(call->args[1], data_sinfo->dtype);
  }

  const auto* attrs = call->attrs.as<ReshapeAttrs>();
  ICHECK_NOTNULL(attrs);
  Array<PrimExpr> new_shape = attrs->new_shape;
  if (!old_shape_prod.defined()) {
    CheckNewShape(call, ctx, new_shape, /*allow_dim_inference=*/false);
    return TensorStructInfo(ShapeExpr(new_shape), data_sinfo->dtype);
  }

  auto [new_shape_prod, dim_to_infer] =
      CheckNewShape(call, ctx, new_shape, /*allow_dim_inference=*/true);

  if (dim_to_infer != -1) {
    PrimExpr inferred_len = analyzer->Simplify(floordiv(old_shape_prod.value(), new_shape_prod));
    new_shape_prod *= inferred_len;
    new_shape.Set(dim_to_infer, inferred_len);
  }

  if (analyzer->CanProve(old_shape_prod.value() != new_shape_prod)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Reshape expects the new shape to be convertible from the old shape. "
                        "However, the old shape is "
                     << data_sinfo->shape << ", with product " << old_shape_prod.value()
                     << ", while the new shape is " << new_shape << ", with product "
                     << new_shape_prod);
  }

  return TensorStructInfo(ShapeExpr(new_shape), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.reshape")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("new_shape", "Shape", "The input new shape.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReshape);

/* relax.transpose */
TVM_REGISTER_NODE_TYPE(TransposeAttrs);

Expr MakeTranspose(Expr data, Optional<Array<Integer>> axes) {
  ObjectPtr<TransposeAttrs> attrs = make_object<TransposeAttrs>();
  attrs->axes = std::move(axes);

  static const Op& op = Op::Get("relax.transpose");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.transpose").set_body_typed(MakeTranspose);

bool IsIdentityPermutation(const std::vector<int>& permutation) {
  for (int i = 0; i < static_cast<int>(permutation.size()); ++i) {
    if (permutation[i] != i) {
      return false;
    }
  }
  return true;
}

StructInfo InferStructInfoTranspose(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<TransposeAttrs>();
  if (data_sinfo->IsUnknownNdim()) {
    if (attrs->axes.defined()) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Transpose cannot be performed when the input tensor " << data_sinfo
                       << " ndim is unknown while the given number of axes " << attrs->axes.value()
                       << " is clear. Please use MatchCast to match the input tensor to a specific "
                          "ndim before doing transpose.");
    }
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  }

  if (attrs->axes.defined()) {
    int n_axis = attrs->axes.value().size();
    if (n_axis != data_sinfo->ndim) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Transpose expects the number of input axes to equal the ndim of the "
                          "input tensor. However, the tensor ndim is "
                       << data_sinfo->ndim << " while the given number of axes is " << n_axis);
    }
  }

  std::vector<int> axes;
  if (attrs->axes.defined()) {
    axes = CheckAxesInRangeNonRepetitive(call, ctx, data_sinfo->ndim, attrs->axes.value());
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

TVM_REGISTER_OP("relax.transpose")
    .set_attrs_type<TransposeAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "nD Tensor", "input tensor to be transposed")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTranspose);

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
  std::vector<int> axes = CheckAxesInRangeNonRepetitive(call, ctx, output_ndim, attrs->axis);

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
    std::vector<int> axes =
        CheckAxesInRangeNonRepetitive(call, ctx, data_sinfo->ndim, attrs->axis.value());

    if (!shape_value.defined()) {
      return TensorStructInfo(data_sinfo->dtype, data_sinfo->ndim - axes.size());
    }
    for (int i = 0; i < static_cast<int>(axes.size()); ++i) {
      // When `axis` is given, the dim length at the axes must be static constant integer 1.
      const auto* int_len = shape_value.value()[axes[i]].as<IntImmNode>();
      if (int_len == nullptr || int_len->value != 1) {
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
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim);
  } else if (data_sinfo->ndim == 0) {
    return TensorStructInfo(ShapeExpr({1}), data_sinfo->dtype);
  } else if (data_sinfo->ndim == 1) {
    return data_sinfo;
  }

  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, /*ndim=*/1);
  }
  PrimExpr shape_prod = IntImm(DataType::Int(64), 1);
  for (int i = 0; i < static_cast<int>(data_shape->values.size()); ++i) {
    shape_prod *= data_shape->values[i];
  }
  return TensorStructInfo(ShapeExpr({shape_prod}), data_sinfo->dtype);
}

TVM_REGISTER_OP("relax.flatten")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFlatten);

}  // namespace relax
}  // namespace tvm
