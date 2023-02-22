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
 * \file src/relax/transform/to_train_mode.cc
 * \brief Rewriting some operators to its training mode.
 */
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/transform.h>

#include "../op/nn/nn.h"
#include "utils.h"

namespace tvm {
namespace relax {

class ToTrainModeMutator : public ExprMutator {
 public:
  static Expr Transform(Expr expr) { return ToTrainModeMutator()(expr); }

 private:
  Expr VisitExpr_(const CallNode* call) {
    if (call->op == Op::Get("relax.nn.batch_norm")) {
      auto attrs = call->attrs.as<BatchNormAttrs>();
      if (!attrs->training) {
        // Rewrite training as True
        return batch_norm(call->args[0], call->args[1], call->args[2], call->args[3], call->args[4],
                          attrs->axis, attrs->epsilon, attrs->center, attrs->scale, true,
                          attrs->momentum);
      }
    }
    return GetRef<Call>(call);
  }
};

namespace transform {
Pass ToTrainMode() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(ToTrainModeMutator::Transform(f));
      };
  return CreateFunctionPass(/*pass_function=*/pass_func,  //
                            /*opt_level=*/0,              //
                            /*pass_name=*/"ToTrainMode",  //
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.ToTrainMode").set_body_typed(ToTrainMode);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
