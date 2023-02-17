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
 * \file tvm/relax/transform/legalize_ops.cc
 * \brief Legalize high-level operator calls in Relax functions to call_tir
    with corresponding low-level TIR PrimFuncs.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include "utils.h"

namespace tvm {
namespace relax {

class LegalizeMutator : public ExprMutator {
 public:
  explicit LegalizeMutator(const IRModule& mod, const Optional<Map<String, PackedFunc>>& cmap)
      : ExprMutator(mod), mod_(std::move(mod)), cmap_(std::move(cmap)) {}

  Expr VisitExpr_(const CallNode* call) final {
    Call visited_call = Downcast<Call>(this->VisitExprPostOrder_(call));
    static const auto& legalize_map = Op::GetAttrMap<FLegalize>("FLegalize");
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    auto* op_node = visited_call->op.as<OpNode>();

    if (op_node == nullptr) {  // not a OpNode
      return visited_call;
    }

    if (!std::all_of(visited_call->args.begin(), visited_call->args.end(),
                     [](Expr arg) { return KnowAllShapeValues(GetStructInfo(arg)); }) ||
        !KnowAllShapeValues(GetStructInfo(visited_call))) {  // Not all shape values are known
      return visited_call;
    }

    auto op = GetRef<Op>(op_node);
    FLegalize flegalize;
    bool has_legalize = false;

    // Priority: customize > default.
    // Check if it has customize legalization registered.
    if (cmap_.defined() && cmap_.value().count(op->name)) {
      flegalize = cmap_.value()[op->name];
      has_legalize = true;
    }
    // Check if it has default legalization registered.
    if (!legalize_map.count(op)) {
      if (op != call_tir_op) {
        LOG(WARNING) << "No legalization func for " << op->name << " is found.";
      }
    } else if (!has_legalize) {
      flegalize = legalize_map[op];
      has_legalize = true;
    }

    if (has_legalize) {
      return flegalize(this->builder_, visited_call);
    } else {
      return visited_call;  // No legalization.
    }
  }

  IRModule Transform() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto updated_func = Downcast<Function>(this->VisitExpr(func));
        builder_->UpdateFunction(gv, Downcast<BaseFunc>(updated_func));
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  IRModule mod_;
  Optional<Map<String, PackedFunc>> cmap_;
};  // namespace relax

namespace transform {

Pass LegalizeOps(Optional<Map<String, PackedFunc>> cmap) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return LegalizeMutator(mod, cmap).Transform(); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"LegalizeOps",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.LegalizeOps").set_body_typed(LegalizeOps);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
