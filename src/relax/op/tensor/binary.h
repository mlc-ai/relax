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
 * \file binary.h
 * \brief The functions to make Relax binary operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_BINARY_H_
#define TVM_RELAX_OP_TENSOR_BINARY_H_

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
#define RELAX_REGISTER_BINARY_OP(OpRegName)                        \
  TVM_REGISTER_OP("relax." OpRegName)                              \
      .set_num_inputs(2)                                           \
      .add_argument("lhs", "Tensor", "The left hand side tensor.") \
      .add_argument("rhs", "Tensor", "The right hand side tensor.")

#define RELAX_BINARY_OP_IMPL(OpName, OpRegName)        \
  Expr OpName(Expr lhs, Expr rhs) {                    \
    static const Op& op = Op::Get("relax." OpRegName); \
    return Call(op, {lhs, rhs}, Attrs(), {});          \
  }                                                    \
  TVM_REGISTER_GLOBAL("relax.op." OpRegName).set_body_typed(OpName)

#define RELAX_REGISTER_BINARY_BROADCAST_OP(OpRegName)                                \
  RELAX_REGISTER_BINARY_OP(OpRegName).set_attr<FInferStructInfo>("FInferStructInfo", \
                                                                 InferStructInfoBroadcastArith)

#define RELAX_REGISTER_CMP_OP(OpRegName)                                             \
  RELAX_REGISTER_BINARY_OP(OpRegName).set_attr<FInferStructInfo>("FInferStructInfo", \
                                                                 InferStructInfoBroadcastCMP)

Expr Add(Expr lhs, Expr rhs);

Expr Subtract(Expr lhs, Expr rhs);

Expr Multiply(Expr lhs, Expr rhs);

Expr Divide(Expr lhs, Expr rhs);

Expr FloorDivide(Expr lhs, Expr rhs);

Expr Less(Expr lhs, Expr rhs);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_BINARY_H_
