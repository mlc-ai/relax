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
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(Conv2dAttrs);

RELAX_REGISTER_BINARY_BROADCAST_OP("add")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1);

RELAX_REGISTER_BINARY_BROADCAST_OP("multiply")
    .describe("Elementwise multiply with broadcasting")
    .set_support_level(1);

RELAY_REGISTER_OP("relax.dense")
    .set_num_inputs(2)
    .add_argument("e1", "Expr", "The input expression")
    .add_argument("e2", "Expr", "The input expression")
    .set_attr<FInferShape>("FInferShape", InferShapeDense)
    .set_attr<FInferType>("FInferType", InferTypeDense);

Expr MakeDense(Expr expr1, Expr expr2) {
  static const Op& op = Op::Get("relax.dense");
  return Call(op, {expr1, expr2}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.dense").set_body_typed(MakeDense);

RELAY_REGISTER_OP("relax.conv2d")
    .set_num_inputs(2)
    .add_argument("e1", "Expr", "The input expression")
    .add_argument("e2", "Expr", "The input expression")
    .set_attrs_type<Conv2dAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeConv2d)
    .set_attr<FInferType>("FInferType", InferTypeDense);

Expr MakeConv2d(Expr expr1, Expr expr2, Array<PrimExpr> kernel_size, Array<PrimExpr> stride,
                Array<PrimExpr> padding, Array<PrimExpr> dilation) {
  static const Op& op = Op::Get("relax.conv2d");
  auto attrs = make_object<Conv2dAttrs>();
  attrs->kernel_size = kernel_size;
  attrs->stride = stride;
  attrs->padding = padding;
  attrs->dilation = dilation;
  return Call(op, {expr1, expr2}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.conv2d").set_body_typed(MakeConv2d);

}  // namespace relax
}  // namespace tvm
