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

#ifndef TVM_RELAX_OP_NN_CONVOLUTION_H_
#define TVM_RELAX_OP_NN_CONVOLUTION_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <string>
#include <utility>
#include <vector>

#include "../op_common.h"
namespace tvm {
namespace relax {

template <typename T>
inline Expr MakeConv(Expr data, Expr weight, Array<PrimExpr> strides, Array<PrimExpr> padding,
                     Array<PrimExpr> dilation, int groups, PrimExpr channels,
                     Array<PrimExpr> kernel_size, std::string data_layout,
                     std::string kernel_layout, std::string out_layout, DataType out_dtype,
                     std::string op_name) {
  auto attrs = make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  const Op& op = Op::Get(op_name);
  return Call(op, {data, weight}, Attrs(attrs), {});
}

Optional<Expr> InferShapeConv2d(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Conv2d op should have 2 arguments");
  }
  Expr shape0 = call->args[0]->shape();
  Expr shape1 = call->args[1]->shape();
  auto* s0 = shape0.as<ShapeExprNode>();
  auto* s1 = shape1.as<ShapeExprNode>();
  auto* attrs = call->attrs.as<Conv2DAttrs>();
  if (s0 && s1) {
    std::vector<PrimExpr> output_shape;
    size_t ndim0 = s0->values.size();
    size_t ndim1 = s1->values.size();
    if (ndim0 != 4 || ndim1 != 4) {
      LOG(INFO) << ndim0;
      LOG(INFO) << ndim1;
      diag_ctx.EmitFatal(Diagnostic::Error(call->span)
                         << "The 2 arguments of Conv2d must be 4D Tensors");
    }
    // N
    output_shape.push_back(s0->values[0]);
    // C
    output_shape.push_back(s1->values[0]);
    // H
    output_shape.push_back((s0->values[2] + 2 * attrs->padding[0] -
                            attrs->dilation[0] * (attrs->kernel_size[0] - 1) - 1) /
                               attrs->strides[0] +
                           1);
    // W
    output_shape.push_back((s0->values[3] + 2 * attrs->padding[1] -
                            attrs->dilation[1] * (attrs->kernel_size[1] - 1) - 1) /
                               attrs->strides[1] +
                           1);
    return ShapeExpr(Array<PrimExpr>{output_shape.begin(), output_shape.end()});
  } else {
    return NullOpt;
  }
}

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_NN_CONVOLUTION_H_
