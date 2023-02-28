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

#include <tvm/meta_schedule/extracted_task.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include "../../meta_schedule/module_equality.h"

namespace tvm {
namespace relax {
namespace backend {

using meta_schedule::ModuleEqual;
using meta_schedule::ModuleEquality;
using meta_schedule::ModuleHash;
using tvm::meta_schedule::ExtractedTask;

/*!
 * \brief Extract the Meta-Schedule tuning task from a given IRModule.
 * \note
 *   1. The task extractor is responsible for task deduplication. The
 *   deduplication is achieved by comparing structural hashes of PrimFuncs.
 *   2. For a PrimFunc, the weight of its corresponding task is the number
 *   of times it called by op Call-TIR. Say in an IRModule there are three
 *   PrimFuncs `fn1`, `fn2` and `fn3` sharing the same structural hash.
 *   Suppose `fn1` is called by 5 Call-TIR ops among all Relax function,
 *   `fn2` is called by 3 Call-TIR and `fn3` is called by 5 Call-TIR.
 *   Then we will have a ExtractedTask for all three functions, whose weight
 *   is 5 + 3 + 2 = 10.
 */
class TaskExtractor : public ExprVisitor {
 public:
  static Array<ExtractedTask> ExtractTask(IRModule mod, Target target, String mod_eq_name) {
    auto mod_eq = meta_schedule::ModuleEquality::Create(mod_eq_name);
    TaskExtractor extractor(mod, target, *mod_eq);
    // We go through each Relax function in the module.
    for (const auto& kv : mod->functions) {
      if (const auto* func = kv.second.as<FunctionNode>()) {
        extractor(GetRef<Function>(func));
      }
    }
    return std::move(extractor.tasks_);
  }

 private:
  explicit TaskExtractor(IRModule mod, Target target, const ModuleEquality& mod_eq)
      : mod_(std::move(mod)),
        target_(std::move(target)),
        mod2task_(/*bucket_count*/ 0, ModuleHash(mod_eq), ModuleEqual(mod_eq)) {
    normalize_mod_func_ = runtime::Registry::Get("tvm.meta_schedule.normalize_mod");
    ICHECK(normalize_mod_func_) << "Normalization function is not found.";
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    // TODO(@tvm-team): When we differentiate the call for tir function and packed function,
    // this logic should be changed accordingly.
    if (!call->op.same_as(call_tir_op)) {
      // Since the Relax function is of A-normal form, the arguments of this call cannot be another
      // Calls. And hence we do not need to recurse into this Call.
      return;
    }

    // Do not extract external function
    if (call->args[0].as<ExternFuncNode>()) {
      return;
    }

    const GlobalVar& global_var = Downcast<GlobalVar>(call->args[0]);
    BaseFunc f = mod_->Lookup(global_var);
    if (f.as<ExternFuncNode>()) {
      return;
    }
    const tir::PrimFunc& func = Downcast<tir::PrimFunc>(f);
    IRModule tir_mod = (*normalize_mod_func_)(func);
    auto it = mod2task_.find(tir_mod);
    if (it != mod2task_.end()) {
      it->second->weight += 1;
      return;
    }

    ExtractedTask task(/*task_name=*/global_var->name_hint,  //
                       /*mod=*/tir_mod,                      //
                       /*target=*/target_,                   //
                       /*dispatched=*/{tir_mod},             //
                       /*weight=*/1);
    tasks_.push_back(task);
    mod2task_.emplace(tir_mod, task);
  }

  IRModule mod_;
  Target target_;
  Array<ExtractedTask> tasks_;
  std::unordered_map<IRModule, ExtractedTask, ModuleHash, ModuleEqual> mod2task_;
  const runtime::PackedFunc* normalize_mod_func_;
};

TVM_REGISTER_GLOBAL("relax.backend.MetaScheduleExtractTask")
    .set_body_typed([](IRModule mod, Target target, String mod_eq_name) {
      return TaskExtractor::ExtractTask(std::move(mod), std::move(target), mod_eq_name);
    });

}  // namespace backend
}  // namespace relax
}  // namespace tvm
