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

#include "../op_common.h"

namespace tvm {
namespace relax {

/* relax.nn.relu */
RELAX_REGISTER_UNARY_OP("nn.relu", /*require_float_dtype=*/false);

/* relax.nn.gelu */
RELAX_REGISTER_UNARY_OP("nn.gelu", /*require_float_dtype=*/true);

/* relax.nn.silu */
RELAX_REGISTER_UNARY_OP("nn.silu", /*require_float_dtype=*/true);

/* relax.nn.softmax */
TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

Expr MakeSoftmax(Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("relax.nn.softmax");
  return Call(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.softmax").set_body_typed(MakeSoftmax);

StructInfo InferStructInfoSoftmax(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  if (data_sinfo->IsUnknownNdim()) {
    return data_sinfo;
  }
  if (!data_sinfo->IsUnknownDtype() && !data_sinfo->dtype.is_float()) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Softmax requires the input tensor to have float "
                                                "dtype. However, the given input dtype is "
                                             << data_sinfo->dtype);
  }
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  CheckAxisInRange(call, ctx, data_sinfo->ndim, attrs->axis);

  return data_sinfo;
}

TVM_REGISTER_OP("relax.nn.softmax")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSoftmax);

bool NormCheckDtypeAndShape(const Call& call, const BlockBuilder& ctx,
                            const Array<TensorStructInfo>& input_sinfo, Array<Integer> axes) {
  Op op = Downcast<Op>(call->op);
  int n_input = op->arguments.size();

  TensorStructInfo data_sinfo = input_sinfo[0];

  std::vector<int> axes_non_neg;
  if (!data_sinfo->IsUnknownNdim()) {
    axes_non_neg = CheckAxesInRangeNonRepetitive(call, ctx, data_sinfo->ndim, axes);
  }
  int n_axis = axes.size();
  if (!data_sinfo->IsUnknownDtype() && !data_sinfo->dtype.is_float()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << op << " requires the input data to have float dtype. However, the given data dtype is "
        << data_sinfo->dtype);
  }
  for (int i = 1; i < n_input; ++i) {
    if (input_sinfo[i]->dtype != data_sinfo->dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op
                       << " requires all the input tensors to have the same dtype. However, the "
                       << op->arguments[i]->name << " has dtype " << input_sinfo[i]->dtype
                       << " which is other than the input data's dtype " << data_sinfo->dtype);
    } else if (input_sinfo[i]->ndim != n_axis) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << op << " requires the input " << op->arguments[i]->name
                       << " to have as many dimensions as the length of input axes. However, the "
                          "given one has ndim "
                       << input_sinfo[i]->ndim << ", which is other than the length of axes "
                       << n_axis);
    }
  }

  std::vector<Array<PrimExpr>> axis_lengths;
  axis_lengths.reserve(n_input);
  if (const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>()) {
    std::vector<PrimExpr> lengths;
    lengths.reserve(n_axis);
    for (int d = 0; d < n_axis; ++d) {
      lengths.push_back(data_shape->values[axes_non_neg[d]]);
    }
    axis_lengths.push_back(lengths);
  }
  for (int i = 1; i < n_input; ++i) {
    if (const auto* shape = input_sinfo[i]->shape.as<ShapeExprNode>()) {
      axis_lengths.push_back(shape->values);
    }
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  for (int i = 1; i < static_cast<int>(axis_lengths.size()); ++i) {
    for (int d = 0; d < n_axis; ++d) {
      if (analyzer->CanProve(axis_lengths[0][d] != axis_lengths[i][d])) {
        ctx->ReportFatal(Diagnostic::Error(call)
                         << op
                         << " requires the input gamma, beta, etc., to have size same as the "
                            "lengths of the data on the given axes. However, there exists "
                         << axis_lengths[0] << " and " << axis_lengths[i] << " that are unequal.");
      } else if (!analyzer->CanProveEqual(axis_lengths[0][d], axis_lengths[i][d])) {
        return true;
      }
    }
  }
  return false;
}

/* relax.nn.batch_norm */
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

Expr MakeBatchNorm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,  //
                   int axis, double epsilon, bool center, bool scale) {
  ObjectPtr<BatchNormAttrs> attrs = make_object<BatchNormAttrs>();
  attrs->axis = axis;
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op& op = Op::Get("relax.nn.batch_norm");
  return Call(op,
              {std::move(data), std::move(gamma), std::move(beta), std::move(moving_mean),
               std::move(moving_var)},
              Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.batch_norm").set_body_typed(MakeBatchNorm);

StructInfo InferStructInfoBatchNorm(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_sinfo, {attrs->axis});

  DataType dtype = input_sinfo[0]->dtype;
  if (unknown_shape) {
    return TupleStructInfo({TensorStructInfo(dtype, input_sinfo[0]->ndim),
                            TensorStructInfo(dtype, /*ndim=*/1),
                            TensorStructInfo(dtype, /*ndim=*/1)});
  } else {
    return TupleStructInfo({input_sinfo[0], input_sinfo[3], input_sinfo[4]});
  }
}

TVM_REGISTER_OP("relax.nn.batch_norm")
    .set_attrs_type<BatchNormAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .add_argument("moving_mean", "Tensor", "Running mean of input.")
    .add_argument("moving_var", "Tensor", "Running variance of input.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBatchNorm);

/* relax.nn.layer_norm */
TVM_REGISTER_NODE_TYPE(LayerNormAttrs);

Expr MakeLayerNorm(Expr data, Expr gamma, Expr beta, Array<Integer> axes, double epsilon,
                   bool center, bool scale) {
  ObjectPtr<LayerNormAttrs> attrs = make_object<LayerNormAttrs>();
  attrs->axes = std::move(axes);
  attrs->epsilon = epsilon;
  attrs->center = center;
  attrs->scale = scale;

  static const Op& op = Op::Get("relax.nn.layer_norm");
  return Call(op, {std::move(data), std::move(gamma), std::move(beta)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.layer_norm").set_body_typed(MakeLayerNorm);

StructInfo InferStructInfoLayerNorm(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  bool unknown_shape = NormCheckDtypeAndShape(call, ctx, input_sinfo, attrs->axes);

  return unknown_shape ? TensorStructInfo(input_sinfo[0]->dtype, input_sinfo[0]->ndim)
                       : input_sinfo[0];
}

TVM_REGISTER_OP("relax.nn.layer_norm")
    .set_attrs_type<LayerNormAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoLayerNorm);

/* relax.nn.matmul */
TVM_REGISTER_NODE_TYPE(MatmulAttrs);

Expr MakeMatmul(Expr a, Expr b, DataType out_dtype) {
  ObjectPtr<MatmulAttrs> attrs = make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.nn.matmul");
  return Call(op, {std::move(a), std::move(b)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.matmul").set_body_typed(MakeMatmul);

StructInfo InferStructInfoMatmul(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo lhs_sinfo = input_sinfo[0];
  TensorStructInfo rhs_sinfo = input_sinfo[1];

  const auto* attrs = call->attrs.as<MatmulAttrs>();
  DataType out_dtype = attrs->out_dtype.is_void()
                           ? InferBinaryArithOpOutDtype(call, ctx, lhs_sinfo, rhs_sinfo)
                           : attrs->out_dtype;

  if (lhs_sinfo->IsUnknownNdim() || rhs_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(out_dtype, kUnknownNDim);
  }
  int lhs_ndim = lhs_sinfo->ndim;
  int rhs_ndim = rhs_sinfo->ndim;
  if (lhs_ndim == 0 || rhs_ndim == 0) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Matmul requires both inputs to have at least 1 dimension. However, "
                     << (lhs_ndim == 0 ? "lhs" : "rhs") << " is a 0-rank tensor.");
  }

  int lhs_prepended = 0;
  int rhs_appended = 0;
  if (lhs_ndim == 1) {
    lhs_ndim = 2;
    lhs_prepended = 1;
  }
  if (rhs_ndim == 1) {
    rhs_ndim = 2;
    rhs_appended = 1;
  }
  int output_ndim = std::max(lhs_ndim, rhs_ndim) - lhs_prepended - rhs_appended;

  const auto* lhs_shape = lhs_sinfo->shape.as<ShapeExprNode>();
  const auto* rhs_shape = rhs_sinfo->shape.as<ShapeExprNode>();
  if (lhs_shape == nullptr || rhs_shape == nullptr) {
    return TensorStructInfo(out_dtype, output_ndim);
  }

  Array<PrimExpr> lhs_shape_prefix{lhs_shape->values.begin(),
                                   lhs_shape->values.end() - 2 + lhs_prepended};
  Array<PrimExpr> rhs_shape_prefix{rhs_shape->values.begin(),
                                   rhs_shape->values.end() - 2 + rhs_appended};
  Optional<Array<PrimExpr>> output_shape_prefix =
      InferBinaryBroadcastShape(call, ctx, lhs_shape_prefix, rhs_shape_prefix);
  if (!output_shape_prefix.defined()) {
    return TensorStructInfo(out_dtype, output_ndim);
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  PrimExpr lhs_reduction_length = lhs_shape->values[lhs_sinfo->ndim - 1];
  PrimExpr rhs_reduction_length = rhs_shape->values[rhs_ndim - 2];
  if (analyzer->CanProve(lhs_reduction_length != rhs_reduction_length)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Matmul requires the reduction length of lhs and rhs to be equal. However, "
                        "the reduction lengths of lhs and rhs are "
                     << lhs_reduction_length << " and " << rhs_reduction_length
                     << " respectively.");
  }

  Array<PrimExpr> output_shape = output_shape_prefix.value();
  if (!lhs_prepended) {
    output_shape.push_back(lhs_shape->values[lhs_ndim - 2]);
  }
  if (!rhs_appended) {
    output_shape.push_back(rhs_shape->values[rhs_ndim - 1]);
  }
  ICHECK_EQ(static_cast<int>(output_shape.size()), output_ndim);
  return TensorStructInfo(ShapeExpr(output_shape), out_dtype);
}

TVM_REGISTER_OP("relax.nn.matmul")
    .set_num_inputs(2)
    .add_argument("a", "Tensor", "The left operand of the matmul.")
    .add_argument("b", "Tensor", "The right operand of the matmul.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMatmul);

/* relax.nn.dropout */
TVM_REGISTER_NODE_TYPE(DropoutAttrs);

Expr MakeDropout(Expr data, double rate) {
  ObjectPtr<DropoutAttrs> attrs = make_object<DropoutAttrs>();
  attrs->rate = rate;

  static const Op& op = Op::Get("relax.nn.dropout");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.dropout").set_body_typed(MakeDropout);

StructInfo InferStructInfoDropout(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  return TupleStructInfo({data_sinfo, data_sinfo});
}

TVM_REGISTER_OP("relax.nn.dropout")
    .set_attrs_type<DropoutAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "Input to which dropout will be applied.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoDropout);

// Structure info inference for cross entropy and softmax cross entropy
StructInfo InferStructInfoCrossEntropy(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo pred_sinfo = input_sinfo[0];
  TensorStructInfo tgt_sinfo = input_sinfo[1];

  DataType dtype = InferBinaryArithOpOutDtype(call, ctx, pred_sinfo, tgt_sinfo);

  if (pred_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Cross entropy requires the input prediction to have exactly two "
                        "dimensions. However, the given prediction have "
                     << pred_sinfo->ndim);
  }
  if (tgt_sinfo->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Cross entropy requires the input target to have exactly two "
                        "dimensions. However, the given prediction have "
                     << tgt_sinfo->ndim);
  }

  const auto* pred_shape = pred_sinfo->shape.as<ShapeExprNode>();
  const auto* tgt_shape = tgt_sinfo->shape.as<ShapeExprNode>();
  if (pred_shape == nullptr || tgt_shape == nullptr) {
    return TensorStructInfo(ShapeExpr(Array<PrimExpr>()), dtype);
  }

  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  if (analyzer->CanProve(pred_shape->values[0] != tgt_shape->values[0]) ||
      analyzer->CanProve(pred_shape->values[1] != tgt_shape->values[1])) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Cross entropy requires the input prediction and target tensors to have "
                        "the same shape. However, the prediction has shape "
                     << pred_sinfo->shape << " while the target has shape " << tgt_sinfo->shape);
  }
  return TensorStructInfo(ShapeExpr(Array<PrimExpr>()), dtype);
}

/* relax.nn.cross_entropy */
Expr MakeCrossEntropy(Expr predictions, Expr targets) {
  static const Op& op = Op::Get("relax.nn.cross_entropy");
  return Call(op, {std::move(predictions), std::move(targets)}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.cross_entropy").set_body_typed(MakeCrossEntropy);

TVM_REGISTER_OP("relax.nn.cross_entropy")
    .set_num_inputs(2)
    .add_argument("predictions", "Tensor", "The predictions.")
    .add_argument("targets", "Tensor", "The targets.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCrossEntropy);

/* relax.nn.softmax_cross_entropy */
Expr MakeSoftmaxCrossEntropy(Expr predictions, Expr targets) {
  static const Op& op = Op::Get("relax.nn.softmax_cross_entropy");
  return Call(op, {std::move(predictions), std::move(targets)}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.softmax_cross_entropy").set_body_typed(MakeSoftmaxCrossEntropy);

TVM_REGISTER_OP("relax.nn.softmax_cross_entropy")
    .set_num_inputs(2)
    .add_argument("predictions", "Tensor", "The predictions.")
    .add_argument("targets", "Tensor", "The targets.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCrossEntropy);

}  // namespace relax
}  // namespace tvm
