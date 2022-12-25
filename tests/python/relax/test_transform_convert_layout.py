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

import tvm
from tvm import relax
from tvm.relax.transform import ConvertLayout
from tvm.script.parser import ir as I, relax as R


@I.ir_module
class conv2d:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        # block 0
        gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.transpose(x, axes=[0, 2, 3, 1])
        gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.transpose(w, axes=[0, 2, 3, 1])
        gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
            gv,
            gv1,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=None,
            kernel_size=[3, 3],
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
            out_dtype="float32",
        )
        gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv2, axes=[0, 3, 1, 2])
        return gv3


def test_conv2d():
    @I.ir_module
    class Conv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            return gv

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2d)
    tvm.ir.assert_structural_equal(mod, conv2d)


@I.ir_module
class conv2d_relu:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        # block 0
        gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.transpose(x, axes=[0, 2, 3, 1])
        gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.transpose(w, axes=[0, 2, 3, 1])
        gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
            gv,
            gv1,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=None,
            kernel_size=[3, 3],
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
            out_dtype="float32",
        )
        gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv2, axes=[0, 3, 1, 2])
        gv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.relu(gv3)
        return gv4


def test_conv2d_relu():
    @I.ir_module
    class Conv2dReLU:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dReLU)
    tvm.ir.assert_structural_equal(mod, conv2d_relu)


@tvm.script.ir_module
class relu_conv2d_relu:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        # block 0
        gv: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.relu(x)
        gv1: R.Tensor((2, 28, 28, 3), dtype="float32") = R.transpose(gv, axes=[0, 2, 3, 1])
        gv2: R.Tensor((4, 3, 3, 3), dtype="float32") = R.transpose(w, axes=[0, 2, 3, 1])
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
            gv1,
            gv2,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=None,
            kernel_size=[3, 3],
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
            out_dtype="float32",
        )
        gv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        gv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.relu(gv4)
        return gv5


def test_relu_conv2d_relu():
    @I.ir_module
    class ReLUConv2dReLU:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            x0: R.Tensor((2, 3, 28, 28), "float32") = R.nn.relu(x)
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x0, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(ReLUConv2dReLU)
    tvm.ir.assert_structural_equal(mod, relu_conv2d_relu)


if __name__ == "__main__":
    test_conv2d()
    test_conv2d_relu()
    test_relu_conv2d_relu()
