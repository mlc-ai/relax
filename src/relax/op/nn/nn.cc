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

#include "nn.h"

namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(DenseAttrs);

RELAX_REGISTER_OP("relax.nn.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<DenseAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight", "2D Tensor", "Weight matrix.")
    .set_attr<FInferShape>("FInferShape", InferShapeDense)
    .set_attr<FInferType>("FInferType", InferTypeDense);

Expr MakeDense(Expr data, Expr weight, PrimExpr units, DataType out_dtype) {
  auto attrs = make_object<DenseAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.nn.dense");

  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.dense").set_body_typed(MakeDense);
TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

Expr MakeSoftmax(Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("relax.nn.softmax");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAX_REGISTER_OP("relax.nn.softmax")
    .describe(R"code(Softmax layer.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeUnaryBroadcast)
    .set_attr<FInferType>("FInferType", InferTypeUnaryBroadcast);

TVM_REGISTER_GLOBAL("relax.op.nn.softmax").set_body_typed(MakeSoftmax);

/* relax.nn.relu */
RELAX_REGISTER_UNARY_OP("nn.relu");

/* relax.nn.gelu */
RELAX_REGISTER_UNARY_OP("nn.gelu");

/* relax.nn.silu */
RELAX_REGISTER_UNARY_OP("nn.silu");

RELAX_REGISTER_OP("relax.nn.flatten")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferShape>("FInferShape", InferShapeFlatten)
    .set_attr<FInferType>("FInferType", InferTypeFlatten);

Expr MakeFlatten(Expr data) {
  static const Op& op = Op::Get("relax.nn.flatten");
  return Call(op, {data}, {}, {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.flatten").set_body_typed(MakeFlatten);

/* relax.nn.batch_norm */
TVM_REGISTER_NODE_TYPE(BatchNormAttrs);

RELAX_REGISTER_OP("relax.nn.batch_norm")
    .set_attrs_type<BatchNormAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "Input to which batch_norm will be applied.")
    .add_argument("gamma", "Tensor", "The gamma scale factor.")
    .add_argument("beta", "Tensor", "The beta offset factor.")
    .add_argument("moving_mean", "Tensor", "Running mean of input.")
    .add_argument("moving_var", "Tensor", "Running variance of input.")
    .set_attr<FInferShape>("FInferShape", InferShapeBatchNorm)
    .set_attr<FInferType>("FInferType", InferTypeBatchNorm);

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

Optional<Expr> InferShapeBatchNorm(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 5) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "BatchNorm op should have 5 arguments, but only " << call->args.size()
                       << "are get.");
  }

  const auto* data_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* mean_shape = call->args[3]->shape().as<ShapeExprNode>();
  const auto* var_shape = call->args[4]->shape().as<ShapeExprNode>();
  if (data_shape == nullptr || mean_shape == nullptr || var_shape == nullptr) {
    return NullOpt;
  }

  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  const auto* gamma_shape = call->args[1]->shape().as<ShapeExprNode>();
  const auto* beta_shape = call->args[2]->shape().as<ShapeExprNode>();
  if (attrs->scale && gamma_shape == nullptr) {
    return NullOpt;
  }
  if (attrs->center && beta_shape == nullptr) {
    return NullOpt;
  }

  return Tuple(
      {GetRef<ShapeExpr>(data_shape), GetRef<ShapeExpr>(mean_shape), GetRef<ShapeExpr>(var_shape)});
}

Type InferTypeBatchNorm(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 5) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "BatchNorm op should have 5 arguments, but only " << call->args.size()
                       << "are get.");
  }

  const auto* data_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* gamma_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  const auto* beta_type = call->args[2]->checked_type().as<DynTensorTypeNode>();
  const auto* mean_type = call->args[3]->checked_type().as<DynTensorTypeNode>();
  const auto* var_type = call->args[4]->checked_type().as<DynTensorTypeNode>();

  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  int axis = attrs->axis;

  if (data_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (data_type->ndim <= axis) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op axis is " << axis << " while the input data tensor only has "
                       << data_type->ndim << " dimensions. Please make sure `axis` is in range [0, "
                       << data_type->ndim << ").");
  }
  if (mean_type == nullptr) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The op input moving mean should has type DynTensorType, but actually it is "
        << call->args[3]->checked_type()->GetTypeKey()
        << ". Please make sure the input has type DynTensorType.");
  } else if (mean_type->ndim != 1 && mean_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input mean should be a 1-dim tensor, while the actual input mean has "
        << mean_type->ndim << " dimensions.");
  }
  if (var_type == nullptr) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The op input moving variance should has type DynTensorType, but actually it is "
        << call->args[4]->checked_type()->GetTypeKey()
        << ". Please make sure the input has type DynTensorType.");
  } else if (var_type->ndim != 1 && var_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input variance should be a 1-dim tensor, while the actual input variance has "
        << var_type->ndim << " dimensions.");
  }
  if (gamma_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input gamma should has type DynTensorType, but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (gamma_type->ndim != 1 && gamma_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input gamma should be a 1-dim tensor, while the actual input gamma has "
        << gamma_type->ndim << " dimensions.");
  }
  if (beta_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input beta should has type DynTensorType, but actually it is "
                       << call->args[2]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  } else if (beta_type->ndim != 1 && beta_type->ndim != -1) {
    diag_ctx.EmitFatal(
        Diagnostic::Error(call->span)
        << "The input beta should be a 1-dim tensor, while the actual input beta has "
        << beta_type->ndim << " dimensions.");
  }

  return TupleType({GetRef<DynTensorType>(data_type), GetRef<DynTensorType>(mean_type),
                    GetRef<DynTensorType>(var_type)});
  // Todo(ruihang): how to do dtype broadcasting?
}

}  // namespace relax
}  // namespace tvm
