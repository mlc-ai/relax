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

Expr Full(ObjectRef shape, Expr fill_value, DataType dtype);

Expr FullLike(Expr data, Expr fill_value);

Expr Ones(Expr shape, DataType dtype);

Expr OnesLike(Expr e);

Expr Zeros(Expr shape, DataType dtype);

Expr ZerosLike(Expr e);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_CREATE_H_
