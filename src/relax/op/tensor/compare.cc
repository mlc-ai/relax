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
#include "./binary.h"

namespace tvm {
namespace relax {

Type InferTypeBinaryCompareBroadcast(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Binary broadcast op should have 2 arguments");
  }
  Type lhs_type = call->args[0]->checked_type();
  Type rhs_type = call->args[1]->checked_type();
  auto* t0 = lhs_type.as<DynTensorTypeNode>();
  auto* t1 = rhs_type.as<DynTensorTypeNode>();
  if (!t0 || !t1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                       << "Both lhs and rhs should be DynTensor for broadcasting, but got "
                       << lhs_type->GetTypeKey() << " and " << rhs_type->GetTypeKey());
  }
  int output_ndim;
  if (t0->IsUnknownNdim() || t1->IsUnknownNdim()) {
    output_ndim = -1;
  } else {
    output_ndim = std::max(t0->ndim, t1->ndim);
  }
  return DynTensorType(output_ndim, DataType::Bool());
}

#define RELAX_REGISTER_COMPARE_BROADCAST_OP(OpName)                               \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr lhs, Expr rhs) { \
    static const Op& op = Op::Get("relax." OpName);                               \
    return Call(op, {lhs, rhs}, Attrs(), {});                                     \
  });                                                                             \
  RELAX_REGISTER_OP("relax." OpName)                                              \
      .set_num_inputs(2)                                                          \
      .add_argument("lhs", "Tensor", "The left hand side tensor.")                \
      .add_argument("rhs", "Tensor", "The right hand side tensor.")               \
      .set_attr<FInferShape>("FInferShape", InferShapeBinaryBroadcast)            \
      .set_attr<FInferType>("FInferType", InferTypeBinaryCompareBroadcast)

RELAX_REGISTER_COMPARE_BROADCAST_OP("equal").set_support_level(1);
RELAX_REGISTER_COMPARE_BROADCAST_OP("not_equal").set_support_level(1);
RELAX_REGISTER_COMPARE_BROADCAST_OP("greater").set_support_level(1);
RELAX_REGISTER_COMPARE_BROADCAST_OP("greater_equal").set_support_level(1);
RELAX_REGISTER_COMPARE_BROADCAST_OP("less").set_support_level(1);
RELAX_REGISTER_COMPARE_BROADCAST_OP("less_equal").set_support_level(1);

}  // namespace relax
}  // namespace tvm
