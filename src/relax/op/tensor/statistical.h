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

#include <algorithm>
#include <utility>

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
#define RELAX_REGISTER_STATISTICAL_OP_AND_IMPL(OpName)                   \
  Expr OpName(Expr data, Optional<Array<Integer>> axis, bool keepdims) { \
    ObjectPtr<StatisticalAttrs> attrs = make_object<StatisticalAttrs>(); \
    attrs->axis = std::move(axis);                                       \
    attrs->keepdims = keepdims;                                          \
    static const Op& op = Op::Get("relax." #OpName);                     \
    return Call(op, {std::move(data)}, Attrs{attrs}, {});                \
  }                                                                      \
  TVM_REGISTER_GLOBAL("relax.op." #OpName).set_body_typed(OpName);       \
  TVM_REGISTER_OP("relax." #OpName)                                      \
      .set_num_inputs(1)                                                 \
      .add_argument("expr", "Tensor", "The input tensor")                \
      .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoStatistical)

/*!
 * \brief Computes the sum of array elements over given axes.
 * \param data The input data
 * \param axis Axis or axes along which a sum is performed. Being `NullOpt` means to sum all the
 * elements of the input tensor
 * \param keepdims If this is set to True, the axes which are reduced are left in the result as
 * dimensions with size one. With this option, the result will broadcast correctly against the
 * input tensor.
 * \return Expr
 */
Expr sum(Expr data, Optional<Array<Integer>> axis, bool keepdims);

/*! \brief Computes the mean of array elements over given axes. */
Expr mean(Expr data, Optional<Array<Integer>> axis, bool keepdims);

/*! \brief Computes the variance of array elements over given axes. */
Expr variance(Expr data, Optional<Array<Integer>> axis, bool keepdims);

/*! \brief Computes the max of array elements over given axes. */
Expr max(Expr data, Optional<Array<Integer>> axis, bool keepdims);

/*! \brief Computes the min of array elements over given axes. */
Expr min(Expr data, Optional<Array<Integer>> axis, bool keepdims);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_STATISTICAL_H_
