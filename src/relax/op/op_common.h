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
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for Relax ops.
 */
#ifndef TVM_RELAX_OP_OP_COMMON_H_
#define TVM_RELAX_OP_OP_COMMON_H_

#include <tvm/arith/analyzer.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

#include "../transform/infer_layout_utils.h"

namespace tvm {
namespace relax {

bool EqualConstInt(const PrimExpr& lhs, int64_t value);

bool EqualCheck(const PrimExpr& lhs, const PrimExpr& rhs);

/*! Quick helper macro
 * - Expose a positional make function to construct the node.
 * - Register op to the registry.
 *
 * We make the decision to always only expose positional argument.
 * We will do rewrapping in the frontend to support language
 * sugars such as keyword arguments and default value.
 *
 * \param OpName the name of registry.
 */
#define RELAX_REGISTER_OP RELAY_REGISTER_OP

#define RELAX_REGISTER_BINARY_BROADCAST_OP(OpName)                                \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr lhs, Expr rhs) { \
    static const Op& op = Op::Get("relax." OpName);                               \
    return Call(op, {lhs, rhs}, Attrs(), {});                                     \
  });                                                                             \
  RELAX_REGISTER_OP("relax." OpName)                                              \
      .set_num_inputs(2)                                                          \
      .add_argument("lhs", "Tensor", "The left hand side tensor.")                \
      .add_argument("rhs", "Tensor", "The right hand side tensor.")               \
      .set_attr<FInferShape>("FInferShape", InferShapeBinaryBroadcast)            \
      .set_attr<FInferType>("FInferType", InferTypeBinaryBroadcast)

#define RELAX_REGISTER_UNARY_OP(OpName)                               \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr e) { \
    static const Op& op = Op::Get("relax." OpName);                   \
    return Call(op, {e}, Attrs(), {});                                \
  });                                                                 \
  RELAX_REGISTER_OP("relax." OpName)                                  \
      .set_num_inputs(1)                                              \
      .add_argument("e", "Tensor", "The input tensor.")               \
      .set_attr<FInferShape>("FInferShape", InferShapeUnaryBroadcast) \
      .set_attr<FInferType>("FInferType", InferTypeUnaryBroadcast)    \
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)

#define RELAX_REGISTER_REDUCTION_OP(OpName)                                         \
  TVM_REGISTER_GLOBAL("relax.op." OpName)                                           \
      .set_body_typed([](Expr data, Optional<Array<Integer>> axis, bool keepdims) { \
        ObjectPtr<ReduceAttrs> attrs = make_object<ReduceAttrs>();                  \
        attrs->axis = std::move(axis);                                              \
        attrs->keepdims = keepdims;                                                 \
        static const Op& op = Op::Get("relax." OpName);                             \
        return Call(op, {std::move(data)}, Attrs{attrs}, {});                       \
      });                                                                           \
  RELAX_REGISTER_OP("relax." OpName)                                                \
      .set_num_inputs(1)                                                            \
      .add_argument("expr", "Tensor", "The input tensor")                           \
      .set_attr<FInferShape>("FInferShape", InferShapeReduction)                    \
      .set_attr<FInferType>("FInferType", InferTypeReduction)

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_OP_COMMON_H_
