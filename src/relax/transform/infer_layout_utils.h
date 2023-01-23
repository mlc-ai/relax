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
 * \file infer_layout_utils.h
 * \brief Utility functions to alter the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */

#ifndef TVM_RELAX_TRANSFORM_INFER_LAYOUT_UTILS_H_
#define TVM_RELAX_TRANSFORM_INFER_LAYOUT_UTILS_H_

#include <tvm/relax/attrs/create.h>
#include <tvm/relax/attrs/datatype.h>
#include <tvm/relax/attrs/image.h>
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <array>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

using tir::Layout;
using NLayout = NestedMsg<Layout>;

/*!
 * \brief An output structure to hold results from FInferCorrectLayout calls.
 * \tparam input_layouts Inferred input layouts.
 * \tparam output_layouts Inferred output layouts.
 * \tparam new_attrs Updated attributes consistent with inferred layouts.
 */
class InferLayoutOutputNode : public Object {
 public:
  Array<NLayout> input_layouts;
  Array<NLayout> output_layouts;
  Attrs new_attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("input_layouts", &input_layouts);
    v->Visit("output_layouts", &output_layouts);
    v->Visit("new_attrs", &new_attrs);
  }

  TVM_DECLARE_BASE_OBJECT_INFO(InferLayoutOutputNode, Object);

  static constexpr const char* _type_key = "relax.transform.InferLayoutOutput";
};

class InferLayoutOutput : public ObjectRef {
 public:
  InferLayoutOutput(Array<NLayout> input_layouts, Array<NLayout> output_layouts, Attrs new_attrs) {
    auto n = make_object<InferLayoutOutputNode>();
    n->input_layouts = std::move(input_layouts);
    n->output_layouts = std::move(output_layouts);
    n->new_attrs = std::move(new_attrs);
    data_ = n;
  }
  TVM_DEFINE_OBJECT_REF_METHODS(InferLayoutOutput, ObjectRef, InferLayoutOutputNode);
};

struct NLayoutEqual {
  bool operator()(const NLayout& a, const NLayout& b) const {
    auto layout_equal = [](const Layout& a, const Layout& b) {
      if (a.defined() && b.defined()) {
        return a.same_as(b) || a.name() == b.name();
      } else {
        return a.defined() == b.defined();
      }
    };
    return Equal(a, b, layout_equal);
  }
};

using VarLayoutMap = Map<Var, NLayout>;

/*!
 * \brief Layout conversion interface.
 * \param call The call node.
 * \param desired_layouts The desired layouts of the operator.
 * \param var_layout_map The layout of the variables.
 */
using FRelaxInferLayout = runtime::TypedPackedFunc<InferLayoutOutput(
    const Call& call, const Map<String, Array<String>>& desired_layouts,
    const VarLayoutMap& var_layout_map)>;

/*!
 * \brief Initialize a layout given the number of dimensions.
 * \param ndim The number of dimensions.
 * \return The initialized layout.
 */
Layout InitialLayout(int ndim);

/*!
 * \brief Initialize a nested layout given the struct info.
 * \param sinfo The sinfo.
 * \return The initialized nested layout.
 */
NLayout InitialNLayout(const StructInfo& sinfo);

/*!
 * \brief Initialize a nested layout given expression
 * \param sinfo The expr
 * \return The initialized nested layout.
 */
NLayout InitialNLayout(const Expr& expr);

/*!
 * \brief Transpose the input layout like the src layout to the dst layout.
 * \param input The input layout.
 * \param src The source layout.
 * \param dst The destination layout.
 * \return The transposed input layout.
 */
Layout TransposeLike(const Layout& input, const Layout& src, const Layout& dst);

/*!
 * \brief Transpose the input string like the src layout to the dst layout.
 * \param input The input str.
 * \param src The source layout.
 * \param dst The destination layout.
 * \return The transposed input str.
 */
String TransposeStrLike(const String& input, const Layout& src, const Layout& dst);

/*!
 * \brief Find axis in the dst layout. 0 represents the first axis, 1 represents the second axis,
 * etc.
 * \param dst The destination layout.
 * \param axis The axis to be found
 * \return The axis in the dst layout.
 */
int FindAxis(const Layout& dst, int axis);

/*!
 * \brief Get the layout of the expr. The expr must be a Tensor.
 * \param var_layout_map The layout of the variables.
 * \param arg The expr.
 * \return The layout of the expr.
 */
Layout GetLayout(const VarLayoutMap& var_layout_map, const Expr& arg);

/*!
 * \brief Get the nested layout of the expr. The expr must be a nested Tensor.
 * \param var_layout_map The layout of the variables.
 * \param arg The expr.
 * \return The nested layout of the expr.
 */
NLayout GetNLayout(const VarLayoutMap& var_layout_map, const Expr& arg);

/*!
 * \brief Check if the op is not in the desired layout
 * \param call The call node contains the op
 * \param desired_layouts The desired layouts of the operator.
 * \return True if the op is not in the desired layout.
 */
bool NoDesiredLayout(const Call& call, const Map<String, Array<String>>& desired_layouts);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_INFER_LAYOUT_UTILS_H_
