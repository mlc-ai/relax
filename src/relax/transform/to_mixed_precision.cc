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
 * \file src/relax/transform/to_mixed_precision.cc
 * \brief Automatic mixed precision pass.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../op/make_op.h"

namespace tvm {
namespace relax {

enum MixedTypeConversionCategory : int {
  MIXED_PRECISION_ALWAYS = 0,
  MIXED_PRECISION_FOLLOW = 1,
  MIXED_PRECISION_NEVER = 2
};

// Return array is of type : [MixedTypeConversionCategory (int), String, String]
// The fields are          : [ConversionCategory, accumulation_datatype, output_datatype]
// Call is a call node, DataType is the mixed precision type
using FTVMMixedPrecisionConversionType = runtime::TypedPackedFunc<Array<ObjectRef>(
    const Call& call_node, const std::string& target_dtype_str)>;

class ToMixedPrecisionMutator : public ExprMutator {
 public:
  void InitVarMap(const relax::Function& func) {
    for (const auto& param : func->params) {
      if (const auto* type = param->checked_type_.as<DynTensorTypeNode>()) {
        std::unordered_map<std::string, Var> type_var_map;
        type_var_map[runtime::DLDataType2String(type->dtype)] = param;
        var_map_[param] = std::move(type_var_map);
      }
    }
  }

  // Visit the use-site of a defined Var
  Expr VisitExpr_(const VarNode* op) {
    if (const auto* type = op->checked_type_.as<DynTensorTypeNode>()) {
      auto it = var_map_.find(GetRef<Var>(op));
      ICHECK(it != var_map_.end());
      auto type_it = it->second.find(runtime::DLDataType2String(type->dtype));
      if (type_it != it->second.end()) {
        return type_it->second;
      } else {
        return builder_->Emit(MakeCast(GetRef<Var>(op), type->dtype));
      }
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  // Visit the use-site of a defined DataflowVar
  Expr VisitExpr_(const DataflowVarNode* op) {
    if (const auto* type = op->checked_type_.as<DynTensorTypeNode>()) {
      auto it = var_map_.find(GetRef<Var>(op));
      ICHECK(it != var_map_.end());
      auto type_it = it->second.find(runtime::DLDataType2String(type->dtype));
      if (type_it != it->second.end()) {
        return type_it->second;
      } else {
        return builder_->Emit(MakeCast(GetRef<Var>(op), type->dtype));
      }
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    auto emit = [this, binding](const Expr& e, const Var& v, DataType dtype) -> relax::Var {
      relax::Var new_var;
      if (this->builder_->CurrentBlockIsDataFlow() && !binding->var.as<DataflowVarNode>()) {
        new_var = this->builder_->EmitOutput(e);
      } else {
        new_var = this->builder_->Emit(e);
      }
      this->UpdateVarMap(v, dtype, new_var);
      return new_var;
    };

    if (const CallNode* call_node = binding->value.as<CallNode>()) {
      // Only handle the following cases
      // var: DynTensorType() = Call(Op, [args], attr)
      Expr cur_op = call_node->op;
      const OpNode* op_node = cur_op.as<OpNode>();
      if (op_node != nullptr) {
        Op op = Downcast<Op>(cur_op);
        const auto attr_map =
            Op::GetAttrMap<FTVMMixedPrecisionConversionType>("FTVMMixedPrecisionConversionType");
        if (attr_map.count(op)) {
          FTVMMixedPrecisionConversionType func = attr_map[op];
          Array<ObjectRef> op_descriptor =
              func(GetRef<Call>(call_node), DLDataType2String(low_precision_type_));
          ICHECK(op_descriptor.size() == 3)
              << "got the wrong number of returned arguments (expected 3 got "
              << op_descriptor.size() << ") from FTVMMixedPrecisionConversionType for "
              << AsText(op, false);

          int64_t op_conversion_type = Downcast<Integer>(op_descriptor[0])->value;
          MixedTypeConversionCategory category =
              static_cast<MixedTypeConversionCategory>(op_conversion_type);
          DataType accumulation_dtype =
              DataType(String2DLDataType(Downcast<String>(op_descriptor[1])));
          if (category == MIXED_PRECISION_ALWAYS) {
            // Cast inputs to fp16
            std::vector<Expr> new_args;
            CastArgsToType(call_node->args, low_precision_type_, &new_args);
            // Cast output according to out_dtype (if necessary)
            if (accumulation_dtype != low_precision_type_) {
              relax::Var accmulate =
                  emit(relax::Call(call_node->op, new_args, call_node->attrs, call_node->type_args),
                       binding->var, accumulation_dtype);
              relax::Var cast_back = emit(relax::MakeCast(accmulate, low_precision_type_),
                                          binding->var, low_precision_type_);
              return;
            } else {
              relax::Var new_var =
                  emit(relax::Call(call_node->op, new_args, call_node->attrs, call_node->type_args),
                       binding->var, accumulation_dtype);
              return;
            }
          } else if (category == MIXED_PRECISION_FOLLOW) {
            // If all the inputs are fp16 available, we stay fp16, otherwise we cast fp16 inputs to
            // fp32 (if necessary) and outputs fp32
            bool need_cast = false;
            for (const relax::Expr arg : call_node->args) {
              // arg is a tensor
              const relax::VarNode* var_node = arg.as<relax::VarNode>();
              ICHECK(var_node != nullptr);
              auto it = var_map_.find(GetRef<Var>(var_node));
              ICHECK(it != var_map_.end());
              if (it->second.find(runtime::DLDataType2String(low_precision_type_)) ==
                  it->second.end()) {
                need_cast = true;
                break;
              }
            }
            // Cast inputs to fp32/fp16 according to need_cast
            std::vector<Expr> new_args;
            CastArgsToType(call_node->args, need_cast ? full_precision_type_ : low_precision_type_,
                           &new_args);
            relax::Var new_var =
                emit(relax::Call(call_node->op, new_args, call_node->attrs, call_node->type_args),
                     binding->var, need_cast ? full_precision_type_ : low_precision_type_);
            return;
          } else if (category == MIXED_PRECISION_NEVER) {
            // cast inputs to fp32
            std::vector<Expr> new_args;
            CastArgsToType(call_node->args, full_precision_type_, &new_args);
            relax::Var new_var =
                emit(relax::Call(call_node->op, new_args, call_node->attrs, call_node->type_args),
                     binding->var, full_precision_type_);
            return;
          } else {
            LOG(FATAL) << "Unsupported MixedTypeConversionCategory: " << category;
          }
        }
      }
    }

    // Not a call node / Not calling an op / Not calling a supported op
    if (const auto* type = binding->value->checked_type().as<relax::DynTensorTypeNode>()) {
      UpdateVarMap(binding->var, type->dtype, binding->var);
    }
    ExprMutator::VisitBinding_(binding);
  }

 private:
  void CastArgsToType(const Array<Expr>& args, DataType to_type, std::vector<Expr>* new_args) {
    for (const relax::Expr arg : args) {
      // arg is a tensor
      const relax::VarNode* var_node = arg.as<relax::VarNode>();
      ICHECK(var_node != nullptr);
      auto it = var_map_.find(GetRef<Var>(var_node));
      ICHECK(it != var_map_.end());
      auto itt = it->second.find(runtime::DLDataType2String(to_type));
      if (itt == it->second.end()) {
        // the input var is never casted to to_type before
        relax::Var casted_var = builder_->Emit(relax::MakeCast(GetRef<Var>(var_node), to_type));
        new_args->push_back(casted_var);
        UpdateVarMap(GetRef<Var>(var_node), to_type, casted_var);
      } else {
        // the input var is already casted to to_type before
        new_args->push_back(itt->second);
      }
    }
  }

  void UpdateVarMap(const Var& var, DataType from_type, const Var& casted_var) {
    auto it = var_map_.find(var);
    if (it == var_map_.end()) {
      std::unordered_map<std::string, Var> type_var_map;
      type_var_map[runtime::DLDataType2String(from_type)] = casted_var;
      var_map_[var] = type_var_map;
    } else {
      it->second[runtime::DLDataType2String(from_type)] = casted_var;
    }
  }

  DataType low_precision_type_ = DataType(DataType::TypeCode::kFloat, 16, 1);
  DataType full_precision_type_ = DataType(DataType::TypeCode::kFloat, 32, 1);
  std::unordered_map<relax::Var, std::unordered_map<std::string, relax::Var>, ObjectPtrHash,
                     ObjectPtrEqual>
      var_map_;
};  // namespace relax

Expr ToMixedPrecision(const relax::Function& f) {
  ToMixedPrecisionMutator mutator;
  mutator.InitVarMap(f);
  return mutator.VisitExpr(f);
}

namespace transform {

Pass ToMixedPrecision() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ToMixedPrecision(f));
      };
  return CreateFunctionPass(pass_func, 0, "ToMixedPrecision", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ToMixedPrecision").set_body_typed(ToMixedPrecision);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
