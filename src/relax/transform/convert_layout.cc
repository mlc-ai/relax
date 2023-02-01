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
 * \file src/relax/transform/convert_layout.cc
 * \brief Automatic layout conversion pass, especially for axis swapping.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"

namespace tvm {
namespace relax {

using tir::Layout;

/*!
 * \brief Main logic to automatically convert the layout of conv2d. Other ops
 * can adapt to such layout conversion following conv2d accordingly.
 *
 * Structurally speaking, a Relax function is composed of a series of VarBinding and
 * MatchCast. And a specific class of VarBindings is the basic unit we want to rewrite.
 * Formally, they are of the form:
 *
 * var = Call(Op, [args], attrs)
 *
 * where Op is a specific op we want to rewrite, and attrs is the attributes of the op.
 * var and args are all exprs with type Tensor or Tuple of Tensors. They might
 * be vars, constants, or Tuple of vars and constants. We register the layout inference function for
 * each op (FRelaxInferLayout), which accepts the current call, the desired layout of conv2d ops,
 * and the layout map of previous vars. The result of the layout inference function is contained in
 * an InferLayoutOutput object, which contains 3 fields: input_layouts, output_layouts, and attr,
 * which represents the expected input layout, output_layout and converted attrs of the new op call.
 * The rewriter will use these info to convert the layout of inputs and attrs of the op call, and
 * note down the new layout of the output.
 *
 * The desired layout of conv2d ops is a map from the name of the op to the desired layout of the
 * desired feature map, weight and output. For example, if we want to convert the layout of conv2d
 * from NCHW to NHWC, we can set the desired layout of conv2d to be {"conv2d": ["NHWC", "OHWI"]}.
 *
 * The way we represent the layout of a var is a NLayout object, which is a nested tuple of Layout.
 * The incoming layout of the module will be set as the default layout (We use ABCD... as the
 * default) Note that for operators like conv, pool, people typically use NHWC to refer to the axes.
 * But to be generic and support more operators, we use ABCD... to refer to the axes.
 *
 * Note that currently the layout conversion of conv2d only support axis swapping, such as NCHW to
 * NWHC. Packed layout such as NCHW to NCHW4c is not supported now.
 */
class LayoutConvertMutator : public ExprMutator {
 public:
  explicit LayoutConvertMutator(const Map<String, Array<String>>& desired_layouts)
      : desired_layouts_(desired_layouts) {}

  void InitVarMap(const Function& func) {
    for (const auto& param : func->params) {
      if (IsNestedTensor(param)) {
        var_layout_map_[param] = InitialNLayout(param);
      }
    }
  }

 private:
  Array<Integer> LayoutToIntegers(const Layout& layout) {
    Array<Integer> ret;
    LayoutDecision src = InitialLayoutDecision(layout.ndim());
    for (size_t i = 0; i < layout.ndim(); ++i) {
      ret.push_back(Integer(src->layout.IndexOf(layout[i])));
    }
    return ret;
  }

  Expr RewriteExpr(const Expr& expr, const NLayout& to) {
    auto fvisitleaf = [&](const Expr& expr, std::array<NLayout, 2> layouts) -> Expr {
      NLayout from = layouts[0], to = layouts[1];
      if (NLayoutEqual()(from, to)) return expr;
      // If not both from and to are dynamic, then none of them can be dynamic.
      ICHECK(!NLayoutEqual()(from, LayoutDecision::InitUnknownDim()) &&
             !NLayoutEqual()(to, LayoutDecision::InitUnknownDim()))
          << "Cannot convert when exactly one of the layouts is dynamic";
      const auto* tensor = GetStructInfoAs<TensorStructInfoNode>(expr);
      ICHECK(tensor != nullptr) << "Expect a tensor, but got: " << expr;
      Layout axes = TransposeLike(InitialLayoutDecision(tensor->ndim)->layout,
                                  from.LeafValue()->layout, to.LeafValue()->layout);
      return permute_dims(expr, LayoutToIntegers(axes));
    };
    return TransformTupleLeaf<LayoutDecision>(
        VarReplacer::Replace(expr, var_remap_),
        std::array<NLayout, 2>({GetNLayout(var_layout_map_, expr), to}), fvisitleaf);
  }

  Array<Expr> RewriteArgs(const Array<Expr>& args, const Array<NLayout>& to) {
    ICHECK(args.size() == to.size());
    std::vector<Expr> new_args;
    for (size_t i = 0; i < args.size(); ++i) {
      if (IsNestedTensor(args[i])) {
        new_args.push_back(RewriteExpr(args[i], to[i]));
      } else {
        new_args.push_back(args[i]);
      }
    }
    return std::move(new_args);
  }

  void VisitBinding(const Binding& binding) final {
    // Emit the binding
    ExprMutator::VisitBinding(binding);
    // The layout is default to be initial if not rewritten.
    if (IsNestedTensor(binding->var)) {
      if (var_layout_map_.find(binding->var) == var_layout_map_.end()) {
        var_layout_map_[binding->var] = InitialNLayout(binding->var);
      }
    }
  }

  Expr VisitVars_(const Var& var) {
    if (IsNestedTensor(var)) {
      // We encounter a var use outside of Call, we rewrite it to initial layout.
      return RewriteExpr(var, InitialNLayout(var));
    }
    return ExprMutator::VisitExpr_(var.get());
  }

  Expr VisitExpr_(const VarNode* op) final { return VisitVars_(GetRef<Var>(op)); }

  Expr VisitExpr_(const DataflowVarNode* op) final { return VisitVars_(GetRef<Var>(op)); }

  bool HasUnknownDimTensor(const NLayout& nlayout) {
    bool find = false;
    auto fvisit = [&](const LayoutDecision& layout) {
      find = find | (NLayoutEqual()(layout, LayoutDecision::InitUnknownDim()));
    };
    ForEachLeaf<LayoutDecision>(nlayout, fvisit);
    return find;
  }

  bool HasUnknownDimTensor(const Array<Expr>& args) {
    for (const auto& arg : args) {
      if (IsNestedTensor(arg)) {
        if (HasUnknownDimTensor(GetNLayout(var_layout_map_, arg))) {
          return true;
        }
      }
    }
    return false;
  }

  Optional<InferLayoutOutput> GetInferLayoutInfo(const CallNode* call_node,
                                                 const Map<String, Array<String>>& desired_layouts,
                                                 const VarLayoutMap& var_layout_map) {
    const OpNode* op_node = call_node->op.as<OpNode>();
    if (op_node == nullptr) return NullOpt;
    Op op = Downcast<Op>(GetRef<Op>(op_node));
    const auto attr_map = Op::GetAttrMap<FRelaxInferLayout>("FRelaxInferLayout");
    if (attr_map.count(op) && !HasUnknownDimTensor(call_node->args)) {
      // If the op has FRelaxInferLayout, and all the input tensors have known ndim
      FRelaxInferLayout f = attr_map[op];
      return f(GetRef<Call>(call_node), desired_layouts, var_layout_map);
    } else {
      // Otherwise, we use the default policy.
      return NullOpt;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call_node) final {
    Optional<InferLayoutOutput> res =
        GetInferLayoutInfo(call_node, desired_layouts_, var_layout_map_);
    ObjectPtr<CallNode> new_call = make_object<CallNode>(*call_node);
    new_call->struct_info_ = NullOpt;
    if (!res.defined()) {
      // Default policy: use the initial layout.
      std::vector<NLayout> input_layout;
      for (const auto& arg : call_node->args) {
        if (IsNestedTensor(arg)) {
          input_layout.push_back(InitialNLayout(arg));
        } else {
          input_layout.push_back(LayoutDecision::InitUnknownDim());
        }
      }
      Array<Expr> new_args = RewriteArgs(call_node->args, std::move(input_layout));
      new_call->args = std::move(new_args);
      ReEmitBinding(binding, builder_->Normalize(Call(new_call)));
      // update the layout map
      if (IsNestedTensor(binding->var)) {
        var_layout_map_[binding->var] = InitialNLayout(binding->var);
      }
    } else {
      // Convert the layout according to the inferred layout output.
      Array<Expr> new_args = RewriteArgs(call_node->args, res.value()->input_layouts);
      new_call->args = std::move(new_args);
      new_call->attrs = std::move(res.value()->new_attrs);
      ReEmitBinding(binding, builder_->Normalize(Call(new_call)));
      // update the layout map
      if (IsNestedTensor(binding->var)) {
        var_layout_map_[binding->var] = res.value()->output_layouts[0];
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* val) final {
    std::vector<NLayout> input_layout;
    for (const auto& field : val->fields) {
      if (IsNestedTensor(field)) {
        // Use the current realized layout to group the tuple;
        input_layout.push_back(GetNLayout(var_layout_map_, field));
      } else {
        input_layout.push_back(LayoutDecision::InitUnknownDim());
      }
    }
    Array<Expr> new_fields = RewriteArgs(val->fields, std::move(input_layout));
    ReEmitBinding(binding, builder_->Normalize(Tuple(new_fields)));
    // update the layout map
    if (IsNestedTensor(binding->var)) {
      var_layout_map_[binding->var] = input_layout;
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val) final {
    if (IsNestedTensor(val->tuple)) {
      // Use the current realized layout to retrieve the field;
      NLayout input_layout = GetNLayout(var_layout_map_, val->tuple);
      ReEmitBinding(binding, builder_->Normalize(
                                 TupleGetItem(RewriteExpr(val->tuple, input_layout), val->index)));
      // update the layout map
      var_layout_map_[binding->var] = input_layout.NestedArray()[val->index];
    } else {
      ExprMutator::VisitBinding_(binding, val);
    }
  }

  std::unordered_map<Var, NLayout, ObjectPtrHash, ObjectPtrEqual> var_layout_map_;
  Map<String, Array<String>> desired_layouts_;
};

Expr ConvertLayoutPass(const Function& f, Map<String, Array<String>> desired_layouts) {
  LayoutConvertMutator mutator(desired_layouts);
  mutator.InitVarMap(f);
  return mutator.VisitExpr(f);
}

namespace transform {

Pass ConvertLayout(Map<String, Array<String>> desired_layouts) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ConvertLayoutPass(f, desired_layouts));
      };
  return CreateFunctionPass(pass_func, 0, "ConvertLayout", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ConvertLayout").set_body_typed(ConvertLayout);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
