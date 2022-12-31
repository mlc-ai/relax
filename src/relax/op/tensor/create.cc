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
 * \file create.cc
 * \brief Creation operators.
 */

#include "../op_common.h"

namespace tvm {
namespace relax {

/* Initialization operators */
TVM_REGISTER_NODE_TYPE(InitAttrs);

/* relax.full */
Expr MakeFull(Expr fill_value, ObjectRef shape, DataType dtype) {
  Expr shape_in_expr{nullptr};
  if (const auto* expr = shape.as<ExprNode>()) {
    shape_in_expr = GetRef<Expr>(expr);
  } else if (const auto* _array = shape.as<ArrayNode>()) {
    shape_in_expr = ShapeExpr(GetRef<Array<PrimExpr>>(_array));
  } else {
    LOG(FATAL) << "Full only expects the input shape to be either an Expr or an Array of PrimExpr. "
                  "However, the given one is "
               << shape->GetTypeKey();
  }

  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.full");
  return Call(op, {std::move(fill_value), std::move(shape_in_expr)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.full").set_body_typed(MakeFull);

StructInfo InferStructInfoFull(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Full op should have 2 arguments");
  }
  const auto* fill_value_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  if (fill_value_sinfo == nullptr || fill_value_sinfo->ndim != 0) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Full requires the input fill value to be zero rank Tensor. However, the given one is "
        << call->args[0]->struct_info_);
  }
  if (shape_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Full requires the input shape to be a Shape. However, the given one is "
                     << call->args[1]->struct_info_->GetTypeKey());
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  DataType out_dtype = attrs->dtype.is_void() ? fill_value_sinfo->dtype : attrs->dtype;
  return TensorStructInfo(/*shape=*/call->args[1], out_dtype);
}

TVM_REGISTER_OP("relax.full")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(2)
    .add_argument("fill_value", "Tensor", "The scalar tensor, denoting the value to fill.")
    .add_argument("shape", "Shape", "The shape of the created tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFull);

/* relax.full_like */
Expr MakeFullLike(Expr data, Expr fill_value) {
  static const Op& op = Op::Get("relax.full_like");
  return Call(op, {std::move(data), std::move(fill_value)}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.full_like").set_body_typed(MakeFullLike);

StructInfo InferStructInfoFullLike(const Call& call, const BlockBuilder& ctx) {
  Array<TensorStructInfo> input_sinfo = GetInputTensorStructInfo(call, ctx);
  TensorStructInfo data_sinfo = input_sinfo[0];
  TensorStructInfo fill_value_sinfo = input_sinfo[1];
  if (fill_value_sinfo->ndim != 0) {
    ctx->ReportFatal(Diagnostic::Error(call) << "FullLike requires the input fill value to be zero "
                                                "rank Tensor. However, the given one has ndim"
                                             << fill_value_sinfo->ndim);
  }

  return data_sinfo;
}

TVM_REGISTER_OP("relax.full_like")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("fill_value", "Tensor", "The scalar value to fill.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFullLike);

// Structure info inference for Ones and Zeros
StructInfo InferStructInfoOnesZeros(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Ones/Zeros should have 1 argument");
  }

  const auto* shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[0]);
  if (shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "Ones/Zeros requires the input shape to be a Shape. However, the given one is "
        << call->args[0]->struct_info_->GetTypeKey());
  }
  const auto* attrs = call->attrs.as<InitAttrs>();
  return TensorStructInfo(/*shape=*/call->args[0], attrs->dtype);
}

/* relax.ones & relax.ones_like */
Expr MakeOnes(Expr shape, DataType dtype) {
  CHECK(!dtype.is_void()) << "Ones op expects the input dtype not to be void";
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.ones");
  return Call(op, {std::move(shape)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.ones").set_body_typed(MakeOnes);

TVM_REGISTER_OP("relax.ones")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "ShapeExpr", "The shape of the created tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoOnesZeros);

RELAX_REGISTER_UNARY_OP("ones_like", /*require_float_dtype=*/false);

/* relax.zeros & relax.zeros_like */
Expr MakeZeros(Expr shape, DataType dtype) {
  CHECK(!dtype.is_void()) << "Zeros op expects the input dtype not to be void";
  ObjectPtr<InitAttrs> attrs = make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op& op = Op::Get("relax.zeros");
  return Call(op, {std::move(shape)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.zeros").set_body_typed(MakeZeros);

TVM_REGISTER_OP("relax.zeros")
    .set_attrs_type<InitAttrs>()
    .set_num_inputs(1)
    .add_argument("shape", "ShapeExpr", "The shape of the created tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoOnesZeros);

RELAX_REGISTER_UNARY_OP("zeros_like", /*require_float_dtype=*/false);

}  // namespace relax
}  // namespace tvm
