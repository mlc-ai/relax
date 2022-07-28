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

#include "unary.h"

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(UniqueAttrs);
TVM_REGISTER_NODE_TYPE(MaxPool2dAttrs);
RELAY_REGISTER_OP("relax.unique")
    .describe(
        "This operation returns the unique elements and the new index of each item in a given "
        "tensor.")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<UniqueAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeUnique)
    .set_attr<FInferType>("FInferType", InferTypeUnique)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.unique");

Expr MakeUnique(Expr data, bool sorted, bool return_inverse, bool return_counts, int dim) {
  auto attrs = make_object<UniqueAttrs>();
  attrs->sorted = sorted;
  attrs->return_inverse = return_inverse;
  attrs->return_counts = return_counts;
  attrs->dim = dim;
  static const Op& op = Op::Get("unique");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.unique").set_body_typed(MakeUnique);

RELAX_REGISTER_UNARY_OP("softmax");

RELAX_REGISTER_UNARY_OP("relu");

RELAY_REGISTER_OP("relax.flatten")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferShape>("FInferShape", InferShapeFlatten)
    .set_attr<FInferType>("FInferType", InferTypeFlatten);

Expr MakeFlatten(Expr data) {
  static const Op& op = Op::Get("relax.flatten");
  return Call(op, {data}, {}, {});
}
TVM_REGISTER_GLOBAL("relax.op.flatten").set_body_typed(MakeFlatten);

RELAY_REGISTER_OP("relax.max_pool2d")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<MaxPool2dAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeMaxPool2d)
    .set_attr<FInferType>("FInferType", InferTypeSame);

Expr MakeMaxPool2d(Expr data, Array<PrimExpr> kernel_size, Array<PrimExpr> stride,
                   Array<PrimExpr> padding, Array<PrimExpr> dilation) {
  auto attrs = make_object<MaxPool2dAttrs>();
  attrs->kernel_size = kernel_size;
  attrs->stride = stride;
  attrs->padding = padding;
  attrs->dilation = dilation;
  static const Op& op = Op::Get("relax.max_pool2d");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.max_pool2d").set_body_typed(MakeMaxPool2d);

}  // namespace relax
}  // namespace tvm
