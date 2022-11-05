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

#include "training.h"
#include "training_nn.h"

namespace tvm {
namespace relax {

RELAX_REGISTER_UNARY_OP("nn.relu");
RELAX_REGISTER_UNARY_OP("nn.gradrelu_");
RELAX_REGISTER_UNARY_OP("nn.softmax");
RELAX_REGISTER_UNARY_OP("nn.sigmoid");
RELAX_REGISTER_UNARY_OP("nn.tanh");

RELAX_REGISTER_BINARY_OP_BASE("nn.dense", InferShapeDense, InferTypeDense);
RELAX_REGISTER_BINARY_OP_BASE("nn.cross_entropy", InferShapeCrossEntropy, InferTypeCrossEntropy);
RELAX_REGISTER_BINARY_OP_BASE("nn.flatten", InferShapeFlatten, InferTypeFlatten);
RELAX_REGISTER_BINARY_OP_BASE("nn.softmax_cross_entropy", InferShapeCrossEntropy, InferTypeCrossEntropy);
RELAX_REGISTER_BINARY_OP_BASE("nn.matmul", InferShapeMatmul, InferTypeMatmul);

}  // namespace relax
}  // namespace tvm
