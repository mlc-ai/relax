/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "training_tensor.h"
#include "../tensor/transform.h"

namespace tvm {
namespace relax {

RELAX_REGISTER_UNARY_OP_BASE("zeros", InferShapeFull, InferTypeFull);
RELAX_REGISTER_UNARY_OP_BASE("ones", InferShapeFull, InferTypeFull);
RELAX_REGISTER_UNARY_OP("log");
RELAX_REGISTER_UNARY_OP("negative");
RELAX_REGISTER_UNARY_OP("ones_like");
RELAX_REGISTER_UNARY_OP("zeros_like");

RELAX_REGISTER_BINARY_OP_BASE("collapse_sum_like", InferShapeCollapseSumLike, InferTypeCollapseSumLike);

}  // namespace relax
}  // namespace tvm
