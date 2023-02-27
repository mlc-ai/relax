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
 * \file tvm/relax/distributed/struct_info.h
 * \brief Struct info for DTensor (Distributed Tensor)
 */

#ifndef TVM_RELAX_DISTRIBUTED_STRUCT_INFO_H_
#define TVM_RELAX_DISTRIBUTED_STRUCT_INFO_H_

#include <tvm/relax/distributed/global_info.h>
#include <tvm/relax/struct_info.h>
namespace tvm {
namespace relax {
namespace distributed {

/*! \brief Describes how data is distributed in each dimension of the device mesh*/
class PlacementNode : public Object {
 public:
  /*! \brief placement for each dim of device mesh. -1 represents replica, and integer >=0
   * represents sharding dimension on tensor*/
  Array<Integer> dim_placement;
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("dim_placement", &dim_placement); }

  bool SEqualReduce(const PlacementNode* other, SEqualReducer equal) const {
    return equal(dim_placement, other->dim_placement);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(dim_placement); }

  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const char* _type_key = "relax.distributed.Placement";
  TVM_DECLARE_FINAL_OBJECT_INFO(PlacementNode, Object);
};

/*!
 * \brief Managed reference to a Placement.
 * \sa PlacementNode
 */
class Placement : public ObjectRef {
 public:
  TVM_DLL explicit Placement(Array<Integer> dim_placement);
  /*! \brief replica dim is printed as "R" and sharding dim is printed as "S[i]". So a text "S[1]R"
   * can be translated into placement[1, -1]*/
  TVM_DLL explicit Placement(String text_format);
  TVM_DEFINE_OBJECT_REF_METHODS(Placement, ObjectRef, PlacementNode);
};

/*!
 * \brief StructInfo of DTensor (Distributed Tensor).
 */
class DTensorStructInfoNode : public StructInfoNode {
 public:
  /*!
   * \brief The struct info inherited from TensorStructInfo
   */
  TensorStructInfo tensor_sinfo;
  /*!
   * \brief The device mesh of the tensor.
   */
  DeviceMesh device_mesh;
  /*!
   * \brief The placement of the tensor among the device mesh.
   */
  Placement placement;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("device_mesh", &device_mesh);
    v->Visit("placement", &placement);
    v->Visit("tensor_sinfo", &tensor_sinfo);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const DTensorStructInfoNode* other, SEqualReducer equal) const {
    return equal(tensor_sinfo, other->tensor_sinfo) && equal(device_mesh, other->device_mesh) &&
           equal(placement, other->placement);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tensor_sinfo);
    hash_reduce(device_mesh);
    hash_reduce(placement);
  }

  static constexpr const char* _type_key = "relax.DTensorStructInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(DTensorStructInfoNode, StructInfoNode);
};

/*!
 * \brief Managed reference to DTensorStructInfoNode.
 * \sa DTensorStructInfoNode
 */
class DTensorStructInfo : public StructInfo {
 public:
  /*!
   * \brief Construction with device mesh and placement.
   * \param tensor_sinfo The struct info inherited from TensorStructInfo
   * \param device_mesh The device mesh of the tensor.
   * \param placement The placement of the tensor among the device mesh.
   * \param span The span of the AST.
   */
  TVM_DLL DTensorStructInfo(TensorStructInfo tensor_sinfo, DeviceMesh device_mesh,
                            Placement placement, Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DTensorStructInfo, StructInfo, DTensorStructInfoNode);
};

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_STRUCT_INFO_H_
