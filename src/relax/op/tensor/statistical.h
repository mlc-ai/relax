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
 * \file statistical.h
 * \brief The functions to make Relax statistical operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_STATISTICAL_H_
#define TVM_RELAX_OP_TENSOR_STATISTICAL_H_

#include <tvm/relax/attrs/statistical.h>

#include <utility>

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
#define RELAX_REGISTER_STATISTICAL_OP(OpRegName)          \
  TVM_REGISTER_OP("relax." OpRegName)                     \
      .set_num_inputs(1)                                  \
      .add_argument("expr", "Tensor", "The input tensor") \
      .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoStatistical)

#define RELAX_STATISTICAL_OP_IMPL(OpName, OpRegName)                     \
  Expr OpName(Expr data, Optional<Array<Integer>> axis, bool keepdims) { \
    ObjectPtr<StatisticalAttrs> attrs = make_object<StatisticalAttrs>(); \
    attrs->axis = std::move(axis);                                       \
    attrs->keepdims = keepdims;                                          \
    static const Op& op = Op::Get("relax." OpRegName);                   \
    return Call(op, {std::move(data)}, Attrs{attrs}, {});                \
  }                                                                      \
  TVM_REGISTER_GLOBAL("relax.op." OpRegName).set_body_typed(OpName)

Expr Sum(Expr data, Optional<Array<Integer>> axis, bool keepdims);

Expr Mean(Expr data, Optional<Array<Integer>> axis, bool keepdims);

Expr Variance(Expr data, Optional<Array<Integer>> axis, bool keepdims);

Expr Max(Expr data, Optional<Array<Integer>> axis, bool keepdims);

Expr Min(Expr data, Optional<Array<Integer>> axis, bool keepdims);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_STATISTICAL_H_
