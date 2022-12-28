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
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tir/data_layout.h>

namespace tvm {
namespace relax {

using tir::Layout;
using NLayout = NestedMsg<Layout>;

/*
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
        return a.name() == b.name();
      } else {
        return a.defined() == b.defined();
      }
    };
    return Equal(a, b, layout_equal);
  }
};

using LayoutMap = std::unordered_map<NLayout, Var, ObjectPtrHash, NLayoutEqual>;
using VarLayoutMap = std::unordered_map<Var, LayoutMap, ObjectPtrHash, ObjectPtrEqual>;

class VarLayoutMapWrapperNode : public Object {
 public:
  VarLayoutMap inner;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  TVM_DECLARE_BASE_OBJECT_INFO(VarLayoutMapWrapperNode, Object);

  static constexpr const char* _type_key = "relax.transform.VarLayoutMapWrapper";
};

class VarLayoutMapWrapper : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(VarLayoutMapWrapper, ObjectRef, VarLayoutMapWrapperNode);
};

/*!
 * \brief Layout convertion inferface.
 * \param call The call node.
 * \param desired_layouts The desired layouts of the operator.
 * \param var_layout_map The layout of the variables.
 */
using FRelaxInferLayout = runtime::TypedPackedFunc<InferLayoutOutput(
    const Call& call, const Map<String, Array<String>>& desired_layouts,
    VarLayoutMapWrapper var_layout_map)>;

bool IsNLayout(const Type& type);

/*!
 * \brief Initialize a layout given the number of dimensions.
 * \param ndim The number of dimensions.
 * \return The initialized layout.
 */
Layout InitialLayout(size_t ndim);

NLayout InitialNLayout(const Type& type);

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
NLayout GetOneValidNLayout(VarLayoutMapWrapper var_layout_map, const Expr& arg);

InferLayoutOutput InferLayoutConv2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutPool2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutAdaptiveAvgPool2D(const Call& call,
                                               const Map<String, Array<String>>& desired_layouts,
                                               VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutSoftmax(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutBatchNorm(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutUnaryEwise(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutBinaryEwise(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutTernaryEwise(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutReduce(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutTranspose(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutExpandDims(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutSqueeze(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutStridedSlice(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutCumsum(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map);

InferLayoutOutput InferLayoutConcatenate(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         VarLayoutMapWrapper var_layout_map);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORMS_INFER_LAYOUT_UTILS_H_
