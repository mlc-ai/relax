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

#include "transform.h"

#include <unordered_set>

namespace tvm {
namespace relax {

/* relax.transpose */
TVM_REGISTER_NODE_TYPE(TransposeAttrs);

RELAX_REGISTER_OP("relax.transpose")
    .set_attrs_type<TransposeAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "nD Tensor", "input tensor to be transposed")
    .set_attr<FInferShape>("FInferShape", InferShapeTranspose)
    .set_attr<FInferType>("FInferType", InferTypeTranspose);

Expr MakeTranspose(Expr data, Optional<Array<Integer>> axes) {
  ObjectPtr<TransposeAttrs> attrs = make_object<TransposeAttrs>();
  attrs->axes = std::move(axes);

  static const Op& op = Op::Get("relax.transpose");
  return Call(op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_REGISTER_GLOBAL("relax.op.transpose").set_body_typed(MakeTranspose);

Optional<Expr> InferShapeTranspose(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Transpose op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<TransposeAttrs>();
  if (shape == nullptr) {
    return NullOpt;
  }

  int ndim = shape->values.size();
  if (attrs->axes.defined() && ndim != static_cast<int>(attrs->axes.value().size())) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Transpose op expects the input axis indices to be a permutation of 0 to "
                       << ndim - 1 << ". However, the length of the given indices is not " << ndim);
  }

  Array<PrimExpr> out_shape;
  std::unordered_set<int> used_axis;
  out_shape.resize(ndim);
  used_axis.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    int dim = attrs->axes.defined() ? attrs->axes.value()[i]->value : (ndim - i - 1);
    if (dim < 0) {
      dim = ndim + dim;
    }

    if (dim < 0 || dim >= ndim) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Transpose expects all axis indices to be in range [-" << ndim << ", "
                         << ndim << "). However, the given indices on axis " << i << " is " << dim);
    }
    if (used_axis.count(dim)) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Transpose expects all axis indices not to duplicate. However the "
                            "given indices has duplicate "
                         << dim);
    }

    out_shape.Set(i, shape->values[dim]);
    used_axis.insert(dim);
  }
  return ShapeExpr(out_shape);
}

Type InferTypeTranspose(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Transpose op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }
  return GetRef<DynTensorType>(input_type);
}

/* relax.reshape */
RELAX_REGISTER_OP("relax.reshape")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("new_shape", "ShapeExpr", "The input new shape.")
    .set_attr<FInferShape>("FInferShape", InferShapeReshape)
    .set_attr<FInferType>("FInferType", InferTypeReshape);

Expr MakeReshape(Expr data, Expr new_shape) {
  static const Op& op = Op::Get("relax.reshape");
  return Call(op, {std::move(data), std::move(new_shape)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.reshape").set_body_typed(MakeReshape);

Optional<Expr> InferShapeReshape(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Reshape op should have 1 argument");
  }

  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* new_shape = call->args[1].as<ShapeExprNode>();

  // If we have no knowledge on the input data shape or the input new shape, just return the input
  // new shape.
  if (shape == nullptr || new_shape == nullptr) {
    return call->args[1];
  }

  int ndim = shape->values.size();
  PrimExpr shape_prod = tir::make_const(tvm::DataType::Int(32), 1);
  for (int i = 0; i < ndim; ++i) {
    shape_prod = shape_prod * shape->values[i];
  }

  int dim_to_infer = -1;
  int new_ndim = new_shape->values.size();
  PrimExpr new_shape_prod = tir::make_const(tvm::DataType::Int(32), 1);
  tvm::arith::Analyzer ana;
  for (int i = 0; i < new_ndim; ++i) {
    PrimExpr dim_len = new_shape->values[i];
    if (ana.CanProveEqual(dim_len, tir::make_const(tvm::DataType::Int(32), -1))) {
      if (dim_to_infer != -1) {
        diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                           << "Reshape op accepts at most one \"-1\" in the new shape. However, "
                              "the new shape on dimension "
                           << dim_to_infer << " and " << i << " are both \"-1\"");
      }
      dim_to_infer = i;
    } else if (ana.CanProveEqual(dim_len, tir::make_const(tvm::DataType::Int(32), 0))) {
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "Reshape op does not accept \"0\" in the new shape. However, the new "
                            "shape on dimension "
                         << i << " is \"0\"");
    } else {
      new_shape_prod = new_shape_prod * dim_len;
    }
  }

  // Todo(ruihang): need a runtime reshape inference function

  Array<PrimExpr> new_shape_arr = new_shape->values;
  if (dim_to_infer != -1) {
    new_shape_arr.Set(dim_to_infer, shape_prod / new_shape_prod);
  }
  return ShapeExpr(new_shape_arr);
}

Type InferTypeReshape(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Reshape op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input has type DynTensorType.");
  }
  const auto* new_shape_type = call->args[1]->checked_type().as<ShapeTypeNode>();
  if (new_shape_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The new shape of the reshape operator should has type ShapeTypeNode, "
                          "but actually it is "
                       << call->args[1]->checked_type()->GetTypeKey()
                       << ". Please make sure the new shape has type ShapeType.");
  }

  // Todo(ruihang): add ndim to ShapeType
  // return DynTensorType(new_shape_type->ndim, input_type->dtype);

  const auto* new_shape = call->args[1].as<ShapeExprNode>();
  if (new_shape != nullptr) {
    return DynTensorType(new_shape->values.size(), input_type->dtype);
  } else {
    return DynTensorType(-1, input_type->dtype);
  }
}

}  // namespace relax
}  // namespace tvm
