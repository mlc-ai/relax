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
 *  1. be prepended with a prefix "relax.op." as the FFI identifier string for the make function,
 *  2. be prepended with a prefix "relax." as the identifier string in the operator registry.
 */
#define RELAX_REGISTER_BINARY_OP_AND_IMPL(OpName)                  \
  Expr OpName(Expr lhs, Expr rhs) {                                \
    static const Op& op = Op::Get("relax." #OpName);               \
    return Call(op, {lhs, rhs}, Attrs(), {});                      \
  }                                                                \
  TVM_REGISTER_GLOBAL("relax.op." #OpName).set_body_typed(OpName); \
  TVM_REGISTER_OP("relax." #OpName)                                \
      .set_num_inputs(2)                                           \
      .add_argument("lhs", "Tensor", "The left hand side tensor.") \
      .add_argument("rhs", "Tensor", "The right hand side tensor.")

#define RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(OpName)             \
  RELAX_REGISTER_BINARY_OP_AND_IMPL(OpName).set_attr<FInferStructInfo>( \
      "FInferStructInfo", InferStructInfoBroadcastArith)

#define RELAX_REGISTER_CMP_OP_AND_IMPL(OpName)                          \
  RELAX_REGISTER_BINARY_OP_AND_IMPL(OpName).set_attr<FInferStructInfo>( \
      "FInferStructInfo", InferStructInfoBroadcastCMP)

/*! \brief Addition with numpy-style broadcasting. */
Expr add(Expr lhs, Expr rhs);

/*! \brief Subtraction with numpy-style broadcasting. */
Expr subtract(Expr lhs, Expr rhs);

/*! \brief Multiplication with numpy-style broadcasting. */
Expr multiply(Expr lhs, Expr rhs);

/*! \brief Division with numpy-style broadcasting. */
Expr divide(Expr lhs, Expr rhs);

/*! \brief Floor division with numpy-style broadcasting. */
Expr floor_divide(Expr lhs, Expr rhs);

/*! \brief Broadcasted elementwise test for (lhs < rhs). */
Expr less(Expr lhs, Expr rhs);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_BINARY_H_
