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

#include "infer_layout_utils.h"

namespace tvm {
namespace relax {

using tir::IterVar;
using tir::Layout;

std::string InitialLayout(size_t ndim) {
  ICHECK(ndim > 0 && ndim <= 26) << "Only support up to 26 dimensions";
  return Layout("ABCDEFGHIJKLMNOPQRSTUVWXYZ").SubLayout(0, ndim).name();
}

Layout TransposeLike(const Layout& input, const Layout& src, const Layout& dst) {
  ICHECK(src.ndim() == dst.ndim() && input.ndim() == src.ndim())
      << "Layouts must have the same size";
  std::vector<IterVar> axes;
  for (size_t i = 0; i < src.ndim(); ++i) {
    axes.push_back(input->axes[src.IndexOf(dst[i])]);
  }
  return Layout(axes);
}

Layout GetOneValidLayout(VarLayoutMap var_layout_map, const Expr& arg) {
  if (const auto* var = arg.as<VarNode>()) {
    auto it = var_layout_map.find(GetRef<Var>(var));
    if (it != var_layout_map.end()) {
      ICHECK((*it).second.size() >= 1) << "No valid layout for " << (*it).first;
      auto itt = (*it).second.begin();
      return Layout((*itt).first);
    }
  } else if (const auto* constant = arg.as<ConstantNode>()) {
    return InitialLayout(constant->checked_type_.as<DynTensorTypeNode>()->ndim);
  }
  LOG(FATAL) << "Cannot get layout for " << arg;
}

InferLayoutOutput InferLayoutConv2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMap var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr && op_node->name == "relax.nn.conv2d") << "Invalid Call";
  const auto& it = desired_layouts.find("relax.nn.conv2d");
  const auto* data_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* weight_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  ICHECK(data_type != nullptr && weight_type != nullptr) << "Invalid Call";
  ICHECK(data_type->ndim == 4 && weight_type->ndim == 4) << "Invalid Call";
  const auto* attrs = call->attrs.as<Conv2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  Layout data_layout, weight_layout, output_layout;
  ObjectPtr<Conv2DAttrs> new_attrs = make_object<Conv2DAttrs>(*attrs);

  if (it != desired_layouts.end()) {
    // We have a desired layout for conv2d.
    data_layout = TransposeLike(InitialLayout(4), attrs->data_layout, (*it).second[0]);
    weight_layout = TransposeLike(InitialLayout(4), attrs->kernel_layout, (*it).second[1]);
    output_layout = TransposeLike(InitialLayout(4),
                                  attrs->out_layout == "" ? attrs->data_layout : attrs->out_layout,
                                  (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0]);
    new_attrs->data_layout = (*it).second[0];
    new_attrs->kernel_layout = (*it).second[1];
    new_attrs->out_layout = (*it).second.size() == 3 ? (*it).second[2] : (*it).second[0];
  } else {
    // We don't have a desired layout for conv2d.
    // We can just propagate the layout from the input.
    data_layout = GetOneValidLayout(var_layout_map, call->args[0]);
    weight_layout = GetOneValidLayout(var_layout_map, call->args[1]);
    output_layout = data_layout;
    new_attrs->data_layout =
        TransposeLike(attrs->data_layout, InitialLayout(4), data_layout).name();
    new_attrs->kernel_layout =
        TransposeLike(attrs->kernel_layout, InitialLayout(4), weight_layout).name();
    new_attrs->out_layout =
        TransposeLike(attrs->out_layout == "" ? attrs->data_layout : attrs->out_layout,
                      InitialLayout(4), output_layout)
            .name();
  }
  return InferLayoutOutput({data_layout, weight_layout}, {output_layout}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutUnaryEwise(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        VarLayoutMap var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";
  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  return InferLayoutOutput({layout}, {layout}, Attrs(call->attrs));
}

InferLayoutOutput InferLayoutBinaryEwise(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         VarLayoutMap var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 2) << "Invalid Call";
  
  Map<String, Var> lhs_layout_map, rhs_layout_map;
  const auto* lhs = call->args[0].as<VarNode>();
  const auto* rhs = call->args[1].as<VarNode>();
  const auto* lhs_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* rhs_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  ICHECK(lhs_type != nullptr && rhs_type != nullptr) << "Invalid Call";

  if (lhs != nullptr) {
    lhs_layout_map = var_layout_map.at(GetRef<Var>(lhs));
  } else {
    lhs_layout_map.Set(InitialLayout(lhs_type->ndim), Var());
  }
  if (rhs != nullptr) {
    rhs_layout_map = var_layout_map.at(GetRef<Var>(rhs));
  } else {
    rhs_layout_map.Set(InitialLayout(rhs_type->ndim), Var());
  }
  // Find a common layout.
  for (const auto& lhs_iter : lhs_layout_map) {
    for (const auto& rhs_iter : rhs_layout_map) {
      if (lhs_iter.first == rhs_iter.first) {
        Layout lhs_layout = Layout(std::string(lhs_iter.first));
        Layout rhs_layout = Layout(std::string(rhs_iter.first));
        return InferLayoutOutput({lhs_layout, rhs_layout}, {lhs_layout}, Attrs(call->attrs));
      }
    }
  }
  // No common layout found.
  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  return InferLayoutOutput({layout, layout}, {layout}, Attrs(call->attrs));
}

}  // namespace relax
}  // namespace tvm
