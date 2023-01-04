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
 * \file unary.h
 * \brief The functions to make Relax unary operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_UNARY_H_
#define TVM_RELAX_OP_TENSOR_UNARY_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Compute element-wise negative value of data.
 * \param e The input data.
 * \return The computed result.
 */
Expr negative(Expr e);

/*! \brief Compute elementwise sin of data. */
Expr sin(Expr e);

/*! \brief Compute elementwise cos of data. */
Expr cos(Expr e);

/*! \brief Compute elementwise tanh of data. */
Expr tanh(Expr e);

/*! \brief Compute elementwise square root of data. */
Expr sqrt(Expr e);

/*! \brief Compute elementwise natural logarithm of data. */
Expr log(Expr e);

/*! \brief Compute elementwise sigmoid of data. */
Expr sigmoid(Expr e);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_UNARY_H_
