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

bool IsNLayout(const Type& type) {
  if (const auto* tensor_type = type.as<DynTensorTypeNode>()) {
    return !tensor_type->IsUnknownNdim();
  } else if (const auto* tuple_type = type.as<TupleTypeNode>()) {
    for (const auto& field : tuple_type->fields) {
      if (!IsNLayout(field)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

Layout InitialLayout(size_t ndim) {
  ICHECK(ndim > 0 && ndim <= 26) << "Only support up to 26 dimensions";
  return Layout("ABCDEFGHIJKLMNOPQRSTUVWXYZ").SubLayout(0, ndim);
}

NLayout InitialNLayout(const Type& type) {
  if (const auto* tensor_type = type.as<DynTensorTypeNode>()) {
    ICHECK(!tensor_type->IsUnknownNdim()) << "Only support tensor with known rank";
    return NLayout(InitialLayout(tensor_type->ndim));
  } else if (const auto* tuple_type = type.as<TupleTypeNode>()) {
    Array<NLayout> fields;
    for (const auto& field : tuple_type->fields) {
      fields.push_back(InitialNLayout(field));
    }
    return NLayout(fields);
  }
  LOG(FATAL) << "Unsupported type " << type;
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

String TransposeStrLike(const String& input, const Layout& src, const Layout& dst) {
  ICHECK(src.ndim() == dst.ndim() && input.size() == src.ndim())
      << "Layouts must have the same size";
  std::string axes;
  for (size_t i = 0; i < src.ndim(); ++i) {
    axes.push_back(input.at(src.IndexOf(dst[i])));
  }
  return axes;
}

NLayout GetOneValidNLayout(VarLayoutMapWrapper var_layout_map, const Expr& arg) {
  if (const auto* var = arg.as<VarNode>()) {
    auto it = var_layout_map->inner.find(GetRef<Var>(var));
    if (it != var_layout_map->inner.end()) {
      ICHECK((*it).second.size() >= 1) << "No valid layout for " << (*it).first;
      return (*it).second.begin()->first;
    }
  } else if (const auto* constant = arg.as<ConstantNode>()) {
    return InitialNLayout(constant->checked_type());
  }
  LOG(FATAL) << "Cannot get layout for " << arg;
}

Layout GetOneValidLayout(VarLayoutMapWrapper var_layout_map, const Expr& arg) {
  NLayout nlayout = GetOneValidNLayout(var_layout_map, arg);
  ICHECK(nlayout.IsLeaf()) << "Cannot get layout for " << arg;
  return nlayout.LeafValue();
}

InferLayoutOutput InferLayoutConv2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map) {
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

InferLayoutOutput InferLayoutPool2d(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr && op_node->name == "relax.nn.max_pool2d") << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr) << "Invalid Call";
  ICHECK(type->ndim == 4) << "Invalid Call";
  const auto* attrs = call->attrs.as<MaxPool2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  ObjectPtr<MaxPool2DAttrs> new_attrs = make_object<MaxPool2DAttrs>(*attrs);
  new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(4), layout).name();
  new_attrs->out_layout = TransposeLike(attrs->out_layout == "" ? attrs->layout : attrs->out_layout,
                                        InitialLayout(4), layout)
                              .name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutAdaptiveAvgPool2D(const Call& call,
                                               const Map<String, Array<String>>& desired_layouts,
                                               VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr && op_node->name == "relax.nn.adaptive_avg_pool2d") << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr) << "Invalid Call";
  ICHECK(type->ndim == 4) << "Invalid Call";
  const auto* attrs = call->attrs.as<AdaptivePool2DAttrs>();
  ICHECK(attrs) << "Invalid Call";

  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  ObjectPtr<AdaptivePool2DAttrs> new_attrs = make_object<AdaptivePool2DAttrs>(*attrs);
  new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(4), layout).name();
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutSoftmax(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr && op_node->name == "relax.nn.softmax") << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr) << "Invalid Call";
  const auto* attrs = call->attrs.as<SoftmaxAttrs>();
  ICHECK(attrs) << "Invalid Call";

  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  ObjectPtr<SoftmaxAttrs> new_attrs = make_object<SoftmaxAttrs>(*attrs);
  new_attrs->axis = layout.name().find('A' + attrs->axis);
  return InferLayoutOutput({layout}, {layout}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutBatchNorm(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr && op_node->name == "relax.nn.batch_norm") << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 5) << "Invalid Call";
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 5; ++i) {
    const auto* type = call->args[i]->checked_type().as<DynTensorTypeNode>();
    ICHECK(type != nullptr) << "Invalid Call";
    initial_layouts.push_back(InitialLayout(type->ndim));
  }
  const auto* attrs = call->attrs.as<BatchNormAttrs>();
  ICHECK(attrs) << "Invalid Call";

  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  ObjectPtr<BatchNormAttrs> new_attrs = make_object<BatchNormAttrs>(*attrs);
  new_attrs->axis = layout.name().find('A' + attrs->axis);
  return InferLayoutOutput(
      {layout, initial_layouts[1], initial_layouts[2], initial_layouts[3], initial_layouts[4]},
      {{layout, initial_layouts[3], initial_layouts[4]}}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutLayerNorm(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr && op_node->name == "relax.nn.layer_norm") << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 3) << "Invalid Call";
  std::vector<NLayout> initial_layouts;
  for (size_t i = 0; i < 3; ++i) {
    const auto* type = call->args[i]->checked_type().as<DynTensorTypeNode>();
    ICHECK(type != nullptr) << "Invalid Call";
    initial_layouts.push_back(InitialLayout(type->ndim));
  }
  const auto* attrs = call->attrs.as<LayerNormAttrs>();
  ICHECK(attrs) << "Invalid Call";

  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  ObjectPtr<LayerNormAttrs> new_attrs = make_object<LayerNormAttrs>(*attrs);
  std::vector<Integer> new_axis;
  for (const auto& axis : attrs->axis) {
    int axis_int = (axis.IntValue() + layout.ndim()) % layout.ndim();
    new_axis.push_back(layout.name().find('A' + axis_int));
  }
  new_attrs->axis = new_axis;
  return InferLayoutOutput({layout, initial_layouts[1], initial_layouts[2]}, {layout},
                           Attrs(new_attrs));
}

InferLayoutOutput InferLayoutUnaryEwise(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        VarLayoutMapWrapper var_layout_map) {
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
                                         VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 2) << "Invalid Call";
  LayoutMap lhs_layout_map, rhs_layout_map;
  const auto* lhs = call->args[0].as<VarNode>();
  const auto* rhs = call->args[1].as<VarNode>();
  const auto* lhs_type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* rhs_type = call->args[1]->checked_type().as<DynTensorTypeNode>();
  ICHECK(lhs_type != nullptr && rhs_type != nullptr) << "Invalid Call";
  ICHECK(!lhs_type->IsUnknownNdim() && !rhs_type->IsUnknownNdim()) << "Invalid Call";

  if (lhs != nullptr) {
    lhs_layout_map = var_layout_map->inner.at(GetRef<Var>(lhs));
  } else {
    lhs_layout_map.insert({InitialNLayout(lhs->checked_type()), Var()});
  }
  if (rhs != nullptr) {
    rhs_layout_map = var_layout_map->inner.at(GetRef<Var>(rhs));
  } else {
    rhs_layout_map.insert({InitialNLayout(rhs->checked_type()), Var()});
  }
  // Find a common layout.
  for (const auto& lhs_iter : lhs_layout_map) {
    for (const auto& rhs_iter : rhs_layout_map) {
      ICHECK(lhs_iter.first.IsLeaf() && rhs_iter.first.IsLeaf()) << "Invalid Call";
      Layout lhs_layout = lhs_iter.first.LeafValue();
      Layout rhs_layout = rhs_iter.first.LeafValue();
      if (lhs_layout.name() == rhs_layout.name()) {
        return InferLayoutOutput({lhs_layout, rhs_layout}, {lhs_layout}, Attrs(call->attrs));
      }
    }
  }
  // No common layout found.
  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  return InferLayoutOutput({layout, layout}, {layout}, Attrs(call->attrs));
}

InferLayoutOutput InferLayoutTernaryEwise(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 3) << "Invalid Call";

  LayoutMap layout_map[3];
  for (size_t i = 0; i < 3; ++i) {
    const auto* var = call->args[i].as<VarNode>();
    const auto* type = call->args[i]->checked_type().as<DynTensorTypeNode>();
    ICHECK(type != nullptr) << "Invalid Call";
    ICHECK(!type->IsUnknownNdim()) << "Invalid Call";
    if (var != nullptr) {
      layout_map[i] = var_layout_map->inner.at(GetRef<Var>(var));
    } else {
      layout_map[i].insert({InitialNLayout(GetRef<Type>(type)), Var()});
    }
  }

  // Find a common layout.
  for (const auto& iter0 : layout_map[0]) {
    for (const auto& iter1 : layout_map[1]) {
      for (const auto& iter2 : layout_map[2]) {
        ICHECK(iter0.first.IsLeaf() && iter1.first.IsLeaf() && iter2.first.IsLeaf())
            << "Invalid Call";
        Layout layout0 = iter0.first.LeafValue();
        Layout layout1 = iter1.first.LeafValue();
        Layout layout2 = iter2.first.LeafValue();
        if (layout0.name() == layout1.name() && layout0.name() == layout2.name()) {
          return InferLayoutOutput({layout0, layout0, layout0}, {layout0}, Attrs(call->attrs));
        }
      }
    }
  }
  // No common layout found.
  Layout layout = GetOneValidLayout(var_layout_map, call->args[0]);
  return InferLayoutOutput({layout, layout, layout}, {layout}, Attrs(call->attrs));
}

InferLayoutOutput InferLayoutReduce(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<ReduceAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";

  Array<Integer> axis;
  if (attrs->axis.defined()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(type->ndim);
    for (int i = 0; i < type->ndim; ++i) {
      axis.push_back(Integer(i));
    }
  }

  std::string axis_str(type->ndim, '0');
  for (const auto& iter : axis) {
    axis_str[iter->value] = '1';
  }
  for (int i = 0, j = 0; i < type->ndim; ++i) {
    if (axis_str[i] != '1') {
      axis_str[i] = 'A' + j++;
    }
  }

  NLayout nlayout = GetOneValidNLayout(var_layout_map, call->args[0]);
  ICHECK(nlayout.IsLeaf()) << "Invalid Call";
  Layout exisiting_layout = nlayout.LeafValue();
  String new_axis_str = TransposeStrLike(axis_str, InitialLayout(type->ndim), exisiting_layout);
  Array<Integer> new_axis;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '1') {
      new_axis.push_back(Integer(i));
    }
  }
  std::string output_layout = new_axis_str;
  output_layout.erase(std::remove(output_layout.begin(), output_layout.end(), '1'),
                      output_layout.end());

  ObjectPtr<ReduceAttrs> new_attrs = make_object<ReduceAttrs>(*attrs);
  new_attrs->axis = new_axis;
  return InferLayoutOutput({exisiting_layout},
                           {attrs->keepdims ? exisiting_layout : Layout(output_layout)},
                           Attrs(new_attrs));
}

InferLayoutOutput InferLayoutTranspose(const Call& call,
                                       const Map<String, Array<String>>& desired_layouts,
                                       VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<TransposeAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";

  Layout exisiting_layout = GetOneValidLayout(var_layout_map, call->args[0]);
  Array<Integer> order;
  if (attrs->axes.defined()) {
    order = attrs->axes.value();
  } else {
    order.reserve(type->ndim);
    for (int i = 0; i < type->ndim; ++i) {
      order.push_back(Integer(type->ndim - i - 1));
    }
  }
  std::string order_str;
  for (const auto& axis : order) {
    order_str.push_back(axis->value + 'A');
  }
  String new_axes = TransposeStrLike(InitialLayout(type->ndim).name(), exisiting_layout, order_str);
  Array<Integer> new_order;
  for (size_t i = 0; i < new_axes.size(); ++i) {
    new_order.push_back(Integer(new_axes.at(i) - 'A'));
  }
  ObjectPtr<TransposeAttrs> new_attrs = make_object<TransposeAttrs>(*attrs);
  new_attrs->axes = new_order;
  return InferLayoutOutput({exisiting_layout}, {InitialLayout(type->ndim)}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutExpandDims(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<ExpandDimsAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";

  Layout exisiting_layout = GetOneValidLayout(var_layout_map, call->args[0]);
  int ndim = type->ndim;
  int n_new_dim = attrs->axis.size();
  int output_ndim = ndim + n_new_dim;
  std::vector<bool> is_new_dim(output_ndim, false);
  for (const auto& axis : attrs->axis) {
    is_new_dim[(axis->value + output_ndim) % output_ndim] = true;
  }
  std::string new_layout;
  for (int i = 0; i < output_ndim; ++i) {
    if (!is_new_dim[i]) {
      new_layout.push_back('A' + i);
    }
  }
  new_layout = TransposeStrLike(new_layout, InitialLayout(ndim), exisiting_layout);
  std::string output_layout;
  for (int i = 0, j = 0; i < output_ndim; ++i) {
    if (is_new_dim[i]) {
      output_layout.push_back('A' + i);
    } else {
      output_layout.push_back(new_layout.at(j++));
    }
  }
  return InferLayoutOutput({exisiting_layout}, {Layout(output_layout)}, Attrs(call->attrs));
}

InferLayoutOutput InferLayoutSqueeze(const Call& call,
                                     const Map<String, Array<String>>& desired_layouts,
                                     VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<SqueezeAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  const auto* shape = call->args[0]->shape().as<ShapeExprNode>();
  ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";
  ICHECK(shape != nullptr) << "Invalid Call";

  Array<Integer> axis;
  if (attrs->axis.defined()) {
    axis = attrs->axis.value();
  } else {
    axis.reserve(type->ndim);
    for (int i = 0; i < type->ndim; ++i) {
      if (tir::is_one(shape->values[i])) {
        axis.push_back(Integer(i));
      }
    }
  }

  std::string axis_str(type->ndim, '0');
  for (const auto& iter : axis) {
    axis_str[iter->value] = '1';
  }
  for (int i = 0, j = 0; i < type->ndim; ++i) {
    if (axis_str[i] != '1') {
      axis_str[i] = 'A' + j++;
    }
  }

  Layout exisiting_layout = GetOneValidLayout(var_layout_map, call->args[0]);
  String new_axis_str = TransposeStrLike(axis_str, InitialLayout(type->ndim), exisiting_layout);
  Array<Integer> new_axis;
  for (size_t i = 0; i < new_axis_str.size(); ++i) {
    if (new_axis_str.at(i) == '1') {
      new_axis.push_back(Integer(i));
    }
  }
  std::string output_layout = new_axis_str;
  output_layout.erase(std::remove(output_layout.begin(), output_layout.end(), '1'),
                      output_layout.end());

  ObjectPtr<SqueezeAttrs> new_attrs = make_object<SqueezeAttrs>(*attrs);
  new_attrs->axis = new_axis;
  return InferLayoutOutput({exisiting_layout}, {Layout(output_layout)}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutStridedSlice(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";

  Layout exisiting_layout = GetOneValidLayout(var_layout_map, call->args[0]);
  std::string axes;
  if (attrs->axes.defined()) {
    for (const auto& axis : attrs->axes.value()) {
      axes.push_back('A' + axis->value);
    }
  } else {
    for (int i = 0; i < type->ndim; ++i) {
      axes.push_back('A' + i);
    }
  }

  Array<Integer> new_axes;
  for (const auto& axis : axes) {
    new_axes.push_back(Integer(exisiting_layout.name().find(axis)));
  }
  ObjectPtr<StridedSliceAttrs> new_attrs = make_object<StridedSliceAttrs>(*attrs);
  new_attrs->axes = new_axes;
  return InferLayoutOutput({exisiting_layout}, {exisiting_layout}, Attrs(new_attrs));
}

InferLayoutOutput InferLayoutCumsum(const Call& call,
                                    const Map<String, Array<String>>& desired_layouts,
                                    VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<CumsumAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";

  Layout exisiting_layout = GetOneValidLayout(var_layout_map, call->args[0]);
  if (attrs->axis.defined()) {
    int axis = (attrs->axis.value()->value + type->ndim) % type->ndim;
    ObjectPtr<CumsumAttrs> new_attrs = make_object<CumsumAttrs>(*attrs);
    new_attrs->axis = Integer(exisiting_layout.name().find('A' + axis));
    return InferLayoutOutput({exisiting_layout}, {exisiting_layout}, Attrs(new_attrs));
  } else {
    return InferLayoutOutput({InitialLayout(type->ndim)}, {InitialLayout(1)}, Attrs(call->attrs));
  }
}

InferLayoutOutput InferLayoutConcatenate(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_GE(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<ConcatenateAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* tuple_shape = call->args[0]->shape().as<TupleNode>();
  ICHECK(tuple_shape != nullptr) << "Invalid Call";

  int n_tensor = tuple_shape->fields.size();
  ICHECK_GE(n_tensor, 0) << "Invalid Call";
  const auto* args = call->args[0].as<TupleNode>();
  std::vector<LayoutMap> layout_maps;
  for (int i = 0; i < n_tensor; ++i) {
    const auto* type = args->fields[i]->checked_type().as<DynTensorTypeNode>();
    ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";
    const auto* var = args->fields[i].as<VarNode>();
    if (var != nullptr) {
      const auto it = var_layout_map->inner.find(GetRef<Var>(var));
      ICHECK(it != var_layout_map->inner.end()) << "var " << var->vid << " has no layout";
      layout_maps.push_back((*it).second);
    }
  }

  for (const auto it : layout_maps[0]) {
    bool all_same = true;
    for (int i = 1; i < n_tensor; ++i) {
      if (layout_maps[i].find(it.first) == layout_maps[i].end()) {
        all_same = false;
        break;
      }
    }
    if (all_same) {
      // Fina a common layout for all tensors
      NLayout nlayout = it.first;
      ICHECK(nlayout.IsLeaf()) << "Invalid Call";
      Layout common_layout = nlayout.LeafValue();
      Array<NLayout> input_layouts;
      for (int i = 0; i < n_tensor; ++i) {
        input_layouts.push_back(common_layout);
      }
      int axis = attrs->axis.defined() ? attrs->axis.value()->value : 0;
      axis = (axis + common_layout.ndim()) % common_layout.ndim();
      ObjectPtr<ConcatenateAttrs> new_attrs = make_object<ConcatenateAttrs>(*attrs);
      new_attrs->axis = Integer(common_layout.name().find('A' + axis));
      return InferLayoutOutput({NLayout(input_layouts)}, {common_layout}, Attrs(new_attrs));
    }
  }

  // No common layout, use initial layout
  int ndim = args->fields[0]->checked_type().as<DynTensorTypeNode>()->ndim;
  Array<NLayout> input_layouts, output_layouts;
  for (int i = 0; i < n_tensor; ++i) {
    input_layouts.push_back(InitialLayout(ndim));
  }
  output_layouts.push_back(InitialLayout(ndim));
  return InferLayoutOutput({input_layouts}, {output_layouts}, Attrs(call->attrs));
}

InferLayoutOutput InferLayoutSplit(const Call& call,
                                   const Map<String, Array<String>>& desired_layouts,
                                   VarLayoutMapWrapper var_layout_map) {
  const OpNode* op_node = call->op.as<OpNode>();
  ICHECK(op_node != nullptr) << "Invalid Call";
  const auto& it = desired_layouts.find(op_node->name);
  ICHECK(it == desired_layouts.end()) << "Unsupported desired layout for " << op_node->name;
  ICHECK_EQ(call->args.size(), 1) << "Invalid Call";

  const auto* attrs = call->attrs.as<SplitAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";
  const auto* type = call->args[0]->checked_type().as<DynTensorTypeNode>();
  ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Invalid Call";

  Layout exisiting_layout = GetOneValidLayout(var_layout_map, call->args[0]);
  ObjectPtr<SplitAttrs> new_attrs = make_object<SplitAttrs>(*attrs);
  int new_axis = (attrs->axis + type->ndim) % type->ndim;
  new_attrs->axis = exisiting_layout.name().find('A' + new_axis);
  Expr tuple_out = InferShapeSplit(call, DiagnosticContext::Default(IRModule()));
  const auto* tuple_ptr = tuple_out.as<TupleNode>();
  ICHECK(tuple_ptr != nullptr) << "Invalid Call";
  Array<NLayout> tuple_layouts = std::vector<NLayout>(tuple_ptr->fields.size(), exisiting_layout);
  return InferLayoutOutput({exisiting_layout}, {tuple_layouts}, Attrs(new_attrs));
}

}  // namespace relax
}  // namespace tvm
