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
 * \file resize.cc
 * \brief Image resize operators.
 */

#include "resize.h"

namespace tvm {
namespace relax {

/* relax.resize2d */
TVM_REGISTER_NODE_TYPE(Resize2DAttrs);

RELAX_REGISTER_OP("relax.image.resize2d")
    .set_attrs_type<Resize2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferShape>("FInferShape", InferShapeResize2d)
    .set_attr<FInferType>("FInferType", InferTypeResize2d);

Expr MakeResize2D(Expr data, Array<PrimExpr> size, Array<FloatImm> roi, String layout,
                  String method, String coordinate_transformation_mode, String rounding_method,
                  double cubic_alpha, int cubic_exclude, double extrapolation_value) {
  ObjectPtr<Resize2DAttrs> attrs = make_object<Resize2DAttrs>();
  attrs->size = std::move(size);
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = std::move(coordinate_transformation_mode);
  attrs->rounding_method = std::move(rounding_method);
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;

  const static Op& op = Op::Get("relax.image.resize2d");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.resize2d").set_body_typed(MakeResize2D);

Optional<Expr> InferShapeResize2d(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Resize2d op should have 1 argument");
  }

  const auto* input_shape = call->args[0]->shape().as<ShapeExprNode>();
  const auto* attrs = call->attrs.as<Resize2DAttrs>();
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
        << "The resize2d operator expects the input layout to be a string of length 4, containing "
           "letters \"N\", \"C\", \"H\", \"W\". However, the given layout is "
        << attrs->layout);
  }

  Array<PrimExpr> size = attrs->size;
  if (size.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The resize2d operator expects the input size to have exactly two "
                          "elements. However, the given size is "
                       << size << ", which contains " << size.size() << " elements");
  }

  Array<PrimExpr> output_shape;
  output_shape.resize(4);
  output_shape.Set(batch_axis, input_shape->values[batch_axis]);
  output_shape.Set(height_axis, size[0]);
  output_shape.Set(width_axis, size[1]);
  output_shape.Set(channel_axis, input_shape->values[channel_axis]);
  return ShapeExpr(output_shape);
}

Type InferTypeResize2d(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Resize2d op should have 1 argument");
  }

  const auto* input_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  if (input_type == nullptr) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The op input data should has type DynTensorType, but actually it is "
                       << call->args[0]->checked_type()->GetTypeKey()
                       << ". Please make sure the input data has type DynTensorType.");
  }
  if (!input_type->IsUnknownNdim() && input_type->ndim != 4) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "The resize2d operator expects the input data to be a tensor of 4 "
                          "dimensions. However, the given data has "
                       << input_type->ndim << " dimensions");
  }

  return DynTensorType(4, input_type->dtype);
}

}  // namespace relax
}  // namespace tvm
