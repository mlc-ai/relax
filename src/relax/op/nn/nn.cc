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
RELAX_REGISTER_UNARY_OP("nn.relu");

/* relax.nn.gelu */
RELAX_REGISTER_UNARY_OP("nn.gelu");

/* relax.nn.silu */
RELAX_REGISTER_UNARY_OP("nn.silu");

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
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx, /*op_name=*/"Softmax");
  if (data_sinfo->IsUnknownNdim()) {
    return data_sinfo;
  }

  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  int axis = attrs->axis;
  int ndim = data_sinfo->ndim;
  if (axis < -ndim || axis >= ndim) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "The input axis " << axis << " is out of range. The input tensor has "
                     << ndim << " dimensions, so axis should be in range [" << -ndim << ", " << ndim
                     << ").");
  }

  return data_sinfo;
}

TVM_REGISTER_OP("relax.nn.softmax")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoSoftmax);

}  // namespace relax
}  // namespace tvm
