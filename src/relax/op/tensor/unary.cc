/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file unary.cc
 * \brief unary operators.
 */

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Quick helper macro
 * - Expose a make function to construct the node.
 * - Register op to the registry.
 * \param OpName The name of operator to register. The name passed in will
 *  1. be prepended with a prefix "relax.op." as the FFI key string for the make function,
 *  2. be prepended with a prefix "relax." as the key string in the operator registry.
 */
#define RELAX_REGISTER_UNARY_OP(OpName)                               \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr e) { \
    static const Op& op = Op::Get("relax." OpName);                   \
    return Call(op, {e}, Attrs(), {});                                \
  });                                                                 \
  TVM_REGISTER_OP("relax." OpName)                                    \
      .set_num_inputs(1)                                              \
      .add_argument("e", "Tensor", "The input tensor.")               \
      .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnary)

TensorStructInfo GetUnaryInputTensorStructInfo(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call) << "Unary op should have exactly 1 argument");
  }
  const auto* input_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  if (input_sinfo == nullptr) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "The input of unary operator should be Tensor. However, the given input is "
                     << call->args[0]->struct_info_->GetTypeKey());
  }
  return GetRef<TensorStructInfo>(input_sinfo);
}

StructInfo InferStructInfoUnary(const Call& call, const BlockBuilder& ctx) {
  return GetUnaryInputTensorStructInfo(call, ctx);
}

/* relax.negative */
RELAX_REGISTER_UNARY_OP("negative");

/* relax.sin */
RELAX_REGISTER_UNARY_OP("sin");

/* relax.cos */
RELAX_REGISTER_UNARY_OP("cos");

/* relax.tanh */
RELAX_REGISTER_UNARY_OP("tanh");

/* relax.sqrt */
RELAX_REGISTER_UNARY_OP("sqrt");

/* relax.log */
RELAX_REGISTER_UNARY_OP("log");

/* relax.sigmoid */
RELAX_REGISTER_UNARY_OP("sigmoid");

/* relax.unique */
TVM_REGISTER_NODE_TYPE(UniqueAttrs);

Expr MakeUnique(Expr data, bool sorted, bool return_inverse, bool return_counts, int dim) {
  auto attrs = make_object<UniqueAttrs>();
  attrs->sorted = sorted;
  attrs->return_inverse = return_inverse;
  attrs->return_counts = return_counts;
  attrs->dim = dim;
  static const Op& op = Op::Get("relax.unique");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.unique").set_body_typed(MakeUnique);

StructInfo InferStructInfoUnique(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo input_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  const auto* unique_attrs = call->attrs.as<UniqueAttrs>();
  // Only default values of these attributes are supported right now.
  if (unique_attrs->return_counts || unique_attrs->return_inverse || unique_attrs->dim != -1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "support for return_inverse, return_counts, and dim is not implemented");
  }

  return TensorStructInfo(input_sinfo->dtype, /*ndim=*/1);
}

TVM_REGISTER_OP("relax.unique")
    .describe(
        "This operation returns the unique elements and the new index of each item in a given "
        "tensor.")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<UniqueAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnique)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.unique");

}  // namespace relax
}  // namespace tvm
