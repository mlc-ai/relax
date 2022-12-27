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
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include "../op/make_op.h"
#include "infer_layout_utils.h"

namespace tvm {
namespace relax {

using tir::Layout;

class LayoutConvertMutator : public ExprMutator {
 public:
  explicit LayoutConvertMutator(Map<String, Array<String>> desired_layouts)
      : desired_layouts_(desired_layouts) {}

  void InitVarMap(const Function& func) {
    for (const auto& param : func->params) {
      if (const auto* type = param->checked_type().as<DynTensorTypeNode>()) {
        ICHECK(type->ndim > 0) << "Only support tensor with known rank";
        var_layout_map_[param].Set(InitialLayout(type->ndim), param);
      } else {
        other_var_map_[param] = param;
      }
    }
  }

 private:
  Expr VisitVars_(const VarNode* op) {
    if (const auto* type = op->checked_type_.as<DynTensorTypeNode>()) {
      ICHECK(type->ndim > 0) << "Only support tensor with known rank";
      auto it = var_layout_map_.find(GetRef<Var>(op));
      ICHECK(it != var_layout_map_.end()) << "Cannot find var " << GetRef<Var>(op) << " in map";
      Layout target_layout = InitialLayout(type->ndim);
      Layout existing_layout = GetOneValidLayout(var_layout_map_, GetRef<Var>(op));
      auto itt = (*it).second.find(target_layout.name());
      if (itt == (*it).second.end()) {
        // This var does not have the target layout, so we need to convert it.
        Layout axes = TransposeLike(target_layout, existing_layout, target_layout);
        Var converted_var = builder_->Emit(
            MakeTranspose((*it).second[existing_layout.name()], LayoutToIntegers(axes)));
        (*it).second.Set(target_layout.name(), converted_var);
        return converted_var;
      } else {
        // This var already has the target layout
        return (*itt).second;
      }
    }
    auto it = other_var_map_.find(GetRef<Var>(op));
    if (it != other_var_map_.end()) {
      return (*it).second;
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const VarNode* op) final { return VisitVars_(op); }

  Expr VisitExpr_(const DataflowVarNode* op) final { return VisitVars_(op); }

  Array<Integer> LayoutToIntegers(const Layout& layout) {
    Array<Integer> ret;
    Layout src = InitialLayout(layout.ndim());
    for (size_t i = 0; i < layout.ndim(); ++i) {
      ret.push_back(Integer(src.IndexOf(layout[i])));
    }
    return ret;
  }

  void UpdateLayoutMap(const Var& var, const Layout& layout, const Var& converted_var) {
    auto it = var_layout_map_.find(var);
    if (it == var_layout_map_.end()) {
      std::unordered_map<String, Var, ObjectHash, ObjectEqual> layout_map;
      layout_map[layout.name()] = converted_var;
      var_layout_map_[var] = layout_map;
    } else {
      it->second.Set(layout.name(), converted_var);
    }
  }

  // Convert the layout of the input arguments to the desired layout.
  // If input_layouts is not defined, the initial layout of the input arguments will be used.
  size_t TransformArgs(Array<Expr> args, Array<Layout> input_layouts, size_t offset,
                       std::vector<Expr>* new_args) {
    for (size_t i = 0; i < args.size(); ++i) {
      if (const auto* var = args[i].as<VarNode>()) {
        const auto* type = var->checked_type().as<DynTensorTypeNode>();
        ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Only support tensor with known rank";
        Layout target_layout = input_layouts[offset++];
        auto it = var_layout_map_.find(GetRef<Var>(var));
        ICHECK(it != var_layout_map_.end()) << "Cannot find the layout of var: " << var;
        auto itt = it->second.find(target_layout.name());
        if (itt != it->second.end()) {
          // This var already has the desired layout.
          new_args->push_back((*itt).second);
        } else {
          // This var does not have the desired layout.
          // Emit a layout transform op.
          Layout existing_layout = GetOneValidLayout(var_layout_map_, GetRef<Var>(var));
          Var existing_var = it->second[existing_layout.name()];
          Layout axes = TransposeLike(InitialLayout(type->ndim), existing_layout, target_layout);
          Var converted_var = builder_->Emit(MakeTranspose(existing_var, LayoutToIntegers(axes)));
          new_args->push_back(converted_var);
          UpdateLayoutMap(GetRef<Var>(var), target_layout, converted_var);
        }
      } else if (const auto* tuple = args[i].as<TupleNode>()) {
        std::vector<Expr> new_tuple_args;
        offset = TransformArgs(tuple->fields, input_layouts, offset, &new_tuple_args);
        new_args->push_back(Tuple(new_tuple_args));
      } else if (const auto* constant = args[i].as<ConstantNode>()) {
        const auto* type = constant->checked_type().as<DynTensorTypeNode>();
        ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Only support tensor with known rank";
        Layout target_layout = input_layouts[offset++];
        Var converted_const = builder_->Emit(
            MakeTranspose(GetRef<Constant>(constant), LayoutToIntegers(target_layout)));
        new_args->push_back(converted_const);
      } else {
        LOG(FATAL) << "Unsupported argument type: " << args[i]->GetTypeKey();
      }
    }
    return offset;
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    auto emit = [this, binding](Expr e, Var v, Layout layout) -> Var {
      relax::Var new_var;
      if (this->builder_->CurrentBlockIsDataFlow() && !binding->var.as<DataflowVarNode>()) {
        new_var = this->builder_->EmitOutput(e);
      } else {
        new_var = this->builder_->Emit(e);
      }
      if (new_var->checked_type().as<DynTensorTypeNode>()) {
        this->UpdateLayoutMap(v, layout, new_var);
      } else {
        auto it = this->other_var_map_.find(v);
        ICHECK(it == this->other_var_map_.end());
        this->other_var_map_.insert({v, new_var});
      }
      return new_var;
    };

    if (const CallNode* call_node = binding->value.as<CallNode>()) {
      const OpNode* op_node = call_node->op.as<OpNode>();
      if (op_node != nullptr) {
        Op op = Downcast<Op>(GetRef<Op>(op_node));
        const auto infer_layout_map = Op::GetAttrMap<FRelaxInferLayout>("FRelaxInferLayout");
        if (infer_layout_map.count(op)) {
          // Infer the layout convertion from the input layouts and the desired layouts.
          FRelaxInferLayout f = infer_layout_map[op];
          InferLayoutOutput layouts = f(GetRef<Call>(call_node), desired_layouts_, var_layout_map_);
          ICHECK_EQ(layouts->output_layouts.size(), 1) << "Only support single output op for now";
          // Convert the layout of inputs
          std::vector<Expr> new_args;
          TransformArgs(call_node->args, layouts->input_layouts, 0, &new_args);
          // Emit the op with the new inputs.
          emit(Call(call_node->op, new_args, layouts->new_attrs), binding->var,
               layouts->output_layouts[0]);
          return;
        } else {
          // We don't know how to convert the layout of this op.
          // Use the original layout.
          std::vector<Expr> new_args;
          for (const auto& arg : call_node->args) {
            new_args.push_back(VisitExpr(arg));
          }
          const auto* type = binding->var->checked_type().as<DynTensorTypeNode>();
          int ndim;
          if (type != nullptr) {
            ICHECK(!type->IsUnknownNdim()) << "Only support tensor with known rank";
            ndim = type->ndim;
          } else {
            ndim = 1;
          }
          emit(Call(call_node->op, new_args, call_node->attrs), binding->var, InitialLayout(ndim));
        }
      }
    } else {
      ExprMutator::VisitBinding_(binding);
    }
  }

  std::unordered_map<Var, Map<String, Var>, ObjectPtrHash, ObjectPtrEqual> var_layout_map_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> other_var_map_;
  Map<String, Array<String>> desired_layouts_;
};

Expr ConvertLayout(const Function& f, Map<String, Array<String>> desired_layouts) {
  LayoutConvertMutator mutator(desired_layouts);
  mutator.InitVarMap(f);
  return mutator.VisitExpr(f);
}

namespace transform {

Pass ConvertLayoutPass(Map<String, Array<String>> desired_layouts) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ConvertLayout(f, desired_layouts));
      };
  return CreateFunctionPass(pass_func, 0, "ConvertLayout", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ConvertLayout").set_body_typed(ConvertLayoutPass);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
