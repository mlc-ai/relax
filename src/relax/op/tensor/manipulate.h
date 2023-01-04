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
 * \file manipulate.h
 * \brief The functions to make Relax tensor manipulation operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_MANIPULATE_H_
#define TVM_RELAX_OP_TENSOR_MANIPULATE_H_

#include <tvm/relax/attrs/manipulate.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

Expr Reshape(Expr data, ObjectRef shape);

Expr PermuteDims(Expr data, Optional<Array<Integer>> axes);

Expr ExpandDims(Expr data, Array<Integer> axis);

Expr Squeeze(Expr data, Optional<Array<Integer>> axis);

Expr Flatten(Expr data);

Expr Concat(Expr data, Optional<Integer> axis);

Expr Split(Expr data, ObjectRef indices_or_sections, int axis);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_MANIPULATE_H_
