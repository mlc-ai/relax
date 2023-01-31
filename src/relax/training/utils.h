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
 * \file tvm/relax/training/utils.h
 * \brief Utility classes and functions for relax training.
 */
#ifndef TVM_RELAX_TRAINING_UTILS_H_
#define TVM_RELAX_TRAINING_UTILS_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <utility>

namespace tvm {
namespace relax {

/*!
 * \brief Local helper to append a specified loss function after the original function.
 * \note
 * 1. This uitl is dedicated to loss functions, not for general purposes.
 * 2. This util can be replaced if we have Inline pass. It is equivalent to inline a tail call in
 * some sense.
 * \param orig_func The function to be appended to.
 * \param loss_func The loss function.
 * \return The result function after appended.
 */
Function AppendLoss(Function orig_func, Function loss_func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRAINING_UTILS_H_
