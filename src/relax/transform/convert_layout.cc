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

#include "../op/make_op.h"
#include "infer_layout_utils.h"

namespace tvm {
namespace relax {

using tir::Layout;

class LayoutConvertMutator : public ExprMutator {
 public:
  explicit LayoutConvertMutator(Map<String, Array<String>> desired_layouts)
      : desired_layouts_(desired_layouts) {
    ObjectPtr<VarLayoutMapWrapperNode> var_layout_map = make_object<VarLayoutMapWrapperNode>();
    var_layout_map_ = VarLayoutMapWrapper(var_layout_map);
  }

  void InitVarMap(const Function& func) {
    ICHECK(var_layout_map_.defined());
    for (const auto& param : func->params) {
      if (IsNLayout(param->checked_type())) {
        auto it = var_layout_map_->inner.find(param);
        if (it == var_layout_map_->inner.end()) {
          LayoutMap layout_map;
          layout_map[InitialNLayout(param->checked_type())] = param;
          var_layout_map_->inner[param] = std::move(layout_map);
        } else {
          it->second.insert({InitialNLayout(param->checked_type()), param});
        }
      }
    }
  }

 private:
  Array<Integer> LayoutToIntegers(const Layout& layout) {
    Array<Integer> ret;
    Layout src = InitialLayout(layout.ndim());
    for (size_t i = 0; i < layout.ndim(); ++i) {
      ret.push_back(Integer(src.IndexOf(layout[i])));
    }
    return ret;
  }

  void UpdateLayoutMap(const Var& var, const NLayout& layout, const Var& converted_var) {
    auto it = var_layout_map_->inner.find(var);
    if (it == var_layout_map_->inner.end()) {
      LayoutMap layout_map;
      layout_map[layout] = converted_var;
      var_layout_map_->inner[var] = layout_map;
    } else {
      it->second.insert({layout, converted_var});
    }
  }

  // Convert the layout of the input arguments to the desired layout.
  // Note: this function is only used when var does not have the desired layout.
  //       because it will not look up the layout map.
  Var TransformVar(const Var& var, const NLayout& from, const NLayout& to) {
    if (const auto* type = var->checked_type().as<DynTensorTypeNode>()) {
      ICHECK(from.IsLeaf() && to.IsLeaf()) << "Invalid layout for var: " << var;
      Layout axes = TransposeLike(InitialLayout(type->ndim), from.LeafValue(), to.LeafValue());
      Var converted_var = builder_->Emit(MakeTranspose(var, LayoutToIntegers(axes)));
      UpdateLayoutMap(var, to, converted_var);
      return converted_var;
    } else if (const auto* type = var->checked_type().as<TupleTypeNode>()) {
      ICHECK(from.IsNested() && to.IsNested()) << "Invalid layout for var: " << var;
      Array<Expr> fields;
      for (size_t i = 0; i < type->fields.size(); ++i) {
        Var field = builder_->Emit(TupleGetItem(var, i));
        UpdateLayoutMap(field, from.NestedArray()[i], field);
        Var new_filed = TransformVar(field, from.NestedArray()[i], to.NestedArray()[i]);
        fields.push_back(new_filed);
      }
      Var converted_var = builder_->Emit(Tuple(fields));
      UpdateLayoutMap(var, to, converted_var);
      return converted_var;
    } else {
      LOG(FATAL) << "Unsupported type: " << var->checked_type();
      return Var();
    }
  }

  Expr VisitVars_(const VarNode* op) {
    if (IsNLayout(op->checked_type())) {
      auto it = var_layout_map_->inner.find(GetRef<Var>(op));
      ICHECK(it != var_layout_map_->inner.end()) << "Cannot find var " << GetRef<Var>(op);
      NLayout target_layout = InitialNLayout(op->checked_type());
      NLayout existing_layout = GetOneValidNLayout(var_layout_map_, GetRef<Var>(op));
      LayoutMap& layout_map = it->second;
      auto itt = layout_map.find(target_layout);
      if (itt == layout_map.end()) {
        // This var does not have the target layout, so we need to convert it.
        return TransformVar(layout_map[existing_layout], existing_layout, target_layout);
      } else {
        // This var already has the target layout
        return itt->second;
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const VarNode* op) final { return VisitVars_(op); }

  Expr VisitExpr_(const DataflowVarNode* op) final { return VisitVars_(op); }

  // Convert the layout of the input arguments to the desired layout.
  // Note: This function will look up the layout map to find the var with the desired layout.
  void TransformArgs(Array<Expr> args, Array<NLayout> input_layouts, std::vector<Expr>* new_args) {
    ICHECK(args.size() == input_layouts.size()) << "Invalid input layouts";
    for (size_t i = 0; i < args.size(); ++i) {
      if (const auto* var = args[i].as<VarNode>()) {
        NLayout existing_layout = GetOneValidNLayout(var_layout_map_, GetRef<Var>(var));
        auto it = var_layout_map_->inner.find(GetRef<Var>(var));
        ICHECK(it != var_layout_map_->inner.end()) << "Cannot find var " << GetRef<Var>(var);
        LayoutMap& layout_map = it->second;
        auto itt = layout_map.find(input_layouts[i]);
        if (itt == layout_map.end()) {
          // This var does not have the target layout, so we need to convert it.
          itt = layout_map.find(existing_layout);
          ICHECK(itt != layout_map.end()) << "Cannot find layout for var " << GetRef<Var>(var);
          new_args->push_back(TransformVar(itt->second, existing_layout, input_layouts[i]));
        } else {
          // This var already has the target layout
          new_args->push_back(itt->second);
        }
      } else if (const auto* tuple = args[i].as<TupleNode>()) {
        ICHECK(input_layouts[i].IsNested()) << "Invalid layout for tuple: " << args[i];
        std::vector<Expr> new_fields;
        TransformArgs(tuple->fields, input_layouts[i].NestedArray(), &new_fields);
        new_args->push_back(Tuple(new_fields));
      } else if (const auto* constant = args[i].as<ConstantNode>()) {
        const auto* type = constant->checked_type().as<DynTensorTypeNode>();
        ICHECK(type != nullptr && !type->IsUnknownNdim()) << "Only support tensor with known rank";
        ICHECK(input_layouts[i].IsLeaf()) << "Invalid layout for constant: " << args[i];
        Var converted_const = builder_->Emit(MakeTranspose(
            GetRef<Constant>(constant), LayoutToIntegers(input_layouts[i].LeafValue())));
        new_args->push_back(converted_const);
      } else {
        LOG(FATAL) << "Unsupported argument type: " << args[i]->GetTypeKey();
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    auto emit = [this, binding](Expr e, Var v, NLayout layout) -> Var {
      relax::Var new_var;
      if (this->builder_->CurrentBlockIsDataFlow() && !binding->var.as<DataflowVarNode>()) {
        new_var = this->builder_->EmitOutput(e);
      } else {
        new_var = this->builder_->Emit(e);
      }
      if (IsNLayout(v->checked_type())) {
        this->UpdateLayoutMap(v, layout, new_var);
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
          InferLayoutOutput res = f(GetRef<Call>(call_node), desired_layouts_, var_layout_map_);
          ICHECK_EQ(res->output_layouts.size(), 1);
          // Convert the layout of inputs
          std::vector<Expr> new_args;
          TransformArgs(call_node->args, res->input_layouts, &new_args);
          // Emit the op with the new inputs.
          emit(Call(call_node->op, new_args, res->new_attrs), binding->var, res->output_layouts[0]);
          return;
        } else {
          // We don't know how to convert the layout of this op.
          // Use the original layout.
          std::vector<Expr> new_args;
          for (const auto& arg : call_node->args) {
            new_args.push_back(VisitExpr(arg));
          }
          emit(Call(call_node->op, new_args, call_node->attrs), binding->var,
               InitialNLayout(binding->var->checked_type()));
        }
      }
    } else if (const TupleGetItemNode* get_node = binding->value.as<TupleGetItemNode>()) {
      // Convert the layout of the tuple.
      if (const auto* var = get_node->tuple.as<VarNode>()) {
        // The tuple is a var.
        NLayout exisiting_layout = GetOneValidNLayout(var_layout_map_, GetRef<Var>(var));
        auto it = var_layout_map_->inner.find(GetRef<Var>(var));
        ICHECK(it != var_layout_map_->inner.end()) << "Cannot find the layout of var: " << var;
        auto itt = it->second.find(exisiting_layout);
        ICHECK(itt != it->second.end()) << "Cannot find the layout of var: " << var;
        ICHECK(exisiting_layout.IsNested());
        emit(TupleGetItem(itt->second, get_node->index), binding->var,
             exisiting_layout.NestedArray()[get_node->index]);
      } else {
        // The tuple is an expr.
        ExprMutator::VisitBinding_(binding);
      }
    } else {
      ExprMutator::VisitBinding_(binding);
    }
  }

  VarLayoutMapWrapper var_layout_map_;
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
