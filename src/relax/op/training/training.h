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
#ifndef TVM_RELAX_OP_TRAINING_TRANING_H_
#define TVM_RELAX_OP_TRAINING_TRANING_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include "../op_common.h"
#include "../tensor/unary.h"

namespace tvm {
namespace relax {

/* Helper Macros */

#define RELAX_REGISTER_UNARY_OP_BASE(OpName, InferShape, InferType)   \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr e) { \
    static const Op& op = Op::Get("relax." OpName);                   \
    return Call(op, {e}, Attrs(), {});                                \
  });                                                                 \
  RELAY_REGISTER_OP("relax." OpName)                                  \
      .set_num_inputs(1)                                              \
      .add_argument("e", "Tensor", "The input tensor.")               \
      .set_attr<FInferShape>("FInferShape", InferShape) \
      .set_attr<FInferType>("FInferType", InferType)

#define RELAX_REGISTER_BINARY_OP_BASE(OpName, InferShape, InferType)                   \
  TVM_REGISTER_GLOBAL("relax.op." OpName).set_body_typed([](Expr lhs, Expr rhs) { \
    static const Op& op = Op::Get("relax." OpName);                               \
    return Call(op, {lhs, rhs}, Attrs(), {});                                     \
  });                                                                             \
  RELAY_REGISTER_OP("relax." OpName)                                              \
      .set_num_inputs(2)                                                          \
      .add_argument("lhs", "Tensor", "The left hand side tensor.")                \
      .add_argument("rhs", "Tensor", "The right hand side tensor.")               \
      .set_attr<FInferShape>("FInferShape", InferShape)            \
      .set_attr<FInferType>("FInferType", InferType)
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TRAINING_TRANING_H_