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
 * \file utils.h
 * \brief Utility functions to be used in relax passes.
 */

#ifndef TVM_RELAX_TRANSFORM_UTILS_H_
#define TVM_RELAX_TRANSFORM_UTILS_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/data_layout.h>

namespace tvm {
namespace relax {

/*!
 * \brief Check if the given structinfo is a scalar tensor. Now the shape of the tensor sinfo must
 * be ShapeExpr.
 * \param sinfo The structinfo to be checked.
 * \return true if the given structinfo is a scalar tensor.
 */
bool IsScalarTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a scalar tensor. Now the shape of the tensor expr must be
 * ShapeExpr.
 * \param expr The expr to be checked.
 * \return true if the given expr is a scalar tensor.
 */
bool IsScalarTensor(const Expr& expr);

/*!
 * \brief Check if the given structinfo is a nested tensor.
 * \param sinfo The structinfo to be checked.
 * \return true if the given structinfo is a nested tensor.
 */
bool IsNestedTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a nested tensor.
 * \param expr The expr to be checked.
 * \return true if the given expr is a nested tensor.
 */
bool IsNestedTensor(const Expr& expr);

/*!
 * \brief Check if the given structinfo is a nested tensor of floating point dtype.
 * \param sinfo The structinfo to be checked.
 * \return true if the given structinfo is a nested tensor of floating point dtype.
 */
bool IsNestedFloatTensor(const StructInfo& sinfo);

/*!
 * \brief Check if the given expr is a nested tensor of floating point dtype.
 * \param expr The expr to be checked.
 * \return true if the given expr is a nested tensor of floating point dtype.
 */
bool IsNestedFloatTensor(const Expr& expr);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_UTILS_H_
