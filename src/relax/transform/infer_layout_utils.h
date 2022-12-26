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

#ifndef TVM_RELAX_TRANSFORMS_INFER_LAYOUT_UTILS_H_
#define TVM_RELAX_TRANSFORMS_INFER_LAYOUT_UTILS_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/data_layout.h>

namespace tvm {
namespace relax {

using tir::Layout;

/*
 * \brief An output structure to hold results from FInferCorrectLayout calls.
 * \tparam input_layouts Inferred input layouts.
 * \tparam output_layouts Inferred output layouts.
 * \tparam new_attrs Updated attributes consistent with inferred layouts.
 */
class InferLayoutOutputNode : public Object {
 public:
  Array<Layout> input_layouts;
  Array<Layout> output_layouts;
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
  InferLayoutOutput(Array<Layout> input_layouts, Array<Layout> output_layouts, Attrs new_attrs) {
    auto n = make_object<InferLayoutOutputNode>();
    n->input_layouts = std::move(input_layouts);
    n->output_layouts = std::move(output_layouts);
    n->new_attrs = std::move(new_attrs);
    data_ = n;
  }
  TVM_DEFINE_OBJECT_REF_METHODS(InferLayoutOutput, ObjectRef, InferLayoutOutputNode);
};

using VarLayoutMap = Map<relax::Var, Map<String, relax::Var>>;

/*!
 * \brief Layout convertion inferface.
 * \param call The call node.
 * \param desired_layouts The desired layouts of the operator.
 * \param var_layout_map The layout of the variables.
 */
using FRelaxInferLayout = runtime::TypedPackedFunc<InferLayoutOutput(
    const Call& call, const Map<String, Array<String>>& desired_layouts,
    VarLayoutMap var_layout_map)>;

/*!
 * \brief Initialize a layout given the number of dimensions.
 * \param ndim The number of dimensions.
 * \return The initialized layout.
 */
std::string InitialLayout(size_t ndim);

/*!
 * \brief Transpose the input layout like the src layout to the dst layout.
 * \param input The input layout.
 * \param src The source layout.
 * \param dst The destination layout.
 * \return The transposed input layout.
 */
Layout TransposeLike(const Layout& input, const Layout& src, const Layout& dst);

/*!
 * \brief Get one valid layout of the variable.
 * \param var_layout_map The layout of the variables.
 * \param var The variable.
 * \return One valid layout of the variable.
 */
Layout GetOneValidLayout(VarLayoutMap var_layout_map, const Expr& arg);

InferLayoutOutput InferLayoutConv2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutUnaryEwise(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutBinaryEwise(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutTernaryEwise(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutReduce(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutTranspose(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutExpandDims(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutSqueeze(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     VarLayoutMap var_layout_map);

InferLayoutOutput InferLayoutStridedSlice(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          VarLayoutMap var_layout_map);
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORMS_INFER_LAYOUT_UTILS_H_
