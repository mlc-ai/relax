# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import tvm
from tvm import relax
from tvm.relax.transform import ToMixedPrecision, OperatorLegalizer
from tvm.script._parser import ir as I, relax as R, tir as T
import tvm.testing
import tvm.relax.transform.mixed_precision


def test_conv2d():
    @I.ir_module
    class Conv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32")
            return gv

    mod = ToMixedPrecision()(Conv2d)
    print(mod.script())


def test_conv2d_relu():
    @I.ir_module
    class Conv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32")
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.relu(gv)
            return gv2

    mod = ToMixedPrecision()(Conv2d)
    print(mod.script())


def test_relu_conv2d_relu():
    @I.ir_module
    class Conv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            x0: R.Tensor((2, 3, 28, 28), "float32") = R.relu(x)
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.conv2d(
                x0, w, kernel_size=[3, 3], out_dtype="float32")
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.relu(gv)
            return gv2

    mod = ToMixedPrecision()(Conv2d)
    print(mod.script())


test_conv2d()
test_conv2d_relu()
test_relu_conv2d_relu()
