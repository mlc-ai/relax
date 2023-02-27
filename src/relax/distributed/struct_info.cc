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
 * \file src/relax/distributed/struct_info.cc
 * \brief Relax dtensor struct info.
 */

#include <tvm/relax/distributed/struct_info.h>
namespace tvm {
namespace relax {
namespace distributed {

String PlacementNode::ToString() const {
  std::stringstream ss;
  for (size_t i = 0; i < dim_placement.size(); ++i) {
    if (i != 0) {
      ss << ", ";
    }
    if (dim_placement[i] == -1) {
      ss << "R";
    } else {
      ss << "S[" << dim_placement[i] << "]";
    }
  }
  return ss.str();
}

Placement::Placement(Array<Integer> dim_placement) {
  ObjectPtr<PlacementNode> n = make_object<PlacementNode>();
  n->dim_placement = std::move(dim_placement);
  data_ = std::move(n);
}

Placement::Placement(String text_format) {
  Array<Integer> dim_placement;
  std::stringstream ss(text_format);
  while (true) {
    char indicator = 0;
    ss >> indicator;
    if (ss.eof()) {
      break;
    }
    if (indicator == 'R') {
      dim_placement.push_back(-1);
    } else if (indicator == 'S') {
      char lbracket;
      ss >> lbracket;
      CHECK_EQ(lbracket, '[');
      std::string substr;
      getline(ss, substr, ']');
      std::stringstream ss2(substr);
      int dim;
      ss2 >> dim;
      dim_placement.push_back(dim);
      CHECK(ss2.eof()) << "Invalid placement format";
    } else if (indicator == ',') {
      continue;
    } else if (indicator == ' ') {
      continue;
    } else {
      LOG(FATAL) << "Invalid placement format";
    }
  }
  ObjectPtr<PlacementNode> n = make_object<PlacementNode>();
  n->dim_placement = std::move(dim_placement);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PlacementNode);
TVM_REGISTER_GLOBAL("relax.distributed.Placement").set_body_typed([](String text_format) {
  return Placement(text_format);
});

// DTensor
DTensorStructInfo::DTensorStructInfo(TensorStructInfo tensor_sinfo, DeviceMesh device_mesh,
                                     Placement placement, Span span) {
  CHECK_EQ(device_mesh->shape.size(), placement->dim_placement.size())
      << "ValueError: The device mesh and placement must have the same dimension size";
  for (auto axis : placement->dim_placement) {
    CHECK_LT(axis->value, tensor_sinfo->ndim)
        << "ValueError: Sharding dimension should be smaller than tensor ndim";
  }
  ObjectPtr<DTensorStructInfoNode> n = make_object<DTensorStructInfoNode>();
  n->device_mesh = std::move(device_mesh);
  n->placement = std::move(placement);
  n->tensor_sinfo = std::move(tensor_sinfo);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DTensorStructInfoNode);

TVM_REGISTER_GLOBAL("relax.distributed.DTensorStructInfo")
    .set_body_typed([](TensorStructInfo tensor_sinfo, DeviceMesh device_mesh, Placement placement,
                       Span span) {
      return DTensorStructInfo(tensor_sinfo, device_mesh, placement, span);
    });

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
