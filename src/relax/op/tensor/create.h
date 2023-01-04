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
 * \file create.h
 * \brief The functions to make Relax tensor-creation operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_CREATE_H_
#define TVM_RELAX_OP_TENSOR_CREATE_H_

#include <tvm/relax/attrs/create.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Fill array with scalar value.
 * \param shape The shape of the created tensor.
 * \param fill_value The value to fill. Must be a scalar tensor.
 * \param dtype The data type of the created tensor.
 * If dtype is not given, it will by default use the dtype of fill_value.
 * \return The result tensor.
 */
Expr full(ObjectRef shape, Expr fill_value, DataType dtype);

/*!
 * \brief Construct a tensor such that
 * - its shape and dtype is the same as the input data tensor's,
 * - its value is filled with the input scalar fill value.
 * \param data The input tensor, which provides the shape and dtype.
 * \param fill_value The value to fill. Must be a scalar tensor.
 * \return The result tensor.
 */
Expr full_like(Expr data, Expr fill_value);

/*!
 * \brief Construct a tensor of all ones, with the input shape and dtype.
 * \param shape The shape of the created tensor.
 * \param dtype The data type of the created tensor.
 * \return The result tensor.
 */
Expr ones(Expr shape, DataType dtype);

/*!
 * \brief Construct a tensor with all ones, with shape and dtype of the input tensor shape.
 * \param e The input tensor, which provides the shape and dtype.
 * \return The result tensor.
 */
Expr ones_like(Expr e);

/*! \brief Construct a tensor of all zeros, with the input shape and dtype. */
Expr zeros(Expr shape, DataType dtype);

/*! \brief Construct a tensor with all zeros, with shape and dtype of the input tensor shape. */
Expr zeros_like(Expr e);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_CREATE_H_
