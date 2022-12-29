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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv2)
        gv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
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
        gv4: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv3)
        gv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv4, axes=[0, 3, 1, 2])
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


@I.ir_module
class conv2d_relu_tanh:
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv2)
        gv4: R.Tensor((2, 26, 26, 4), dtype="float32") = R.tanh(gv3)
        gv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv4, axes=[0, 3, 1, 2])
        return gv5


def test_conv2d_relu_tanh():
    @I.ir_module
    class Conv2dReLUTanh:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            gv3: R.Tensor((2, 4, 26, 26), "float32") = R.tanh(gv2)
            return gv3

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dReLUTanh)
    tvm.ir.assert_structural_equal(mod, conv2d_relu_tanh)


@I.ir_module
class conv2d_add:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"),
        w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.transpose(bias, axes=[0, 2, 3, 1])
        gv4: R.Tensor((2, 26, 26, 4), dtype="float32") = R.add(gv2, gv3)
        gv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv4, axes=[0, 3, 1, 2])
        return gv5


def test_conv2d_add():
    @I.ir_module
    class Conv2dAdd:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dAdd)
    tvm.ir.assert_structural_equal(mod, conv2d_add)


@I.ir_module
class conv2d_add_relu_conv2d:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"),
        w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.transpose(bias, axes=[0, 2, 3, 1])
        gv4: R.Tensor((2, 26, 26, 4), dtype="float32") = R.add(gv2, gv3)
        gv5: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv4)
        gv6: R.Tensor((2, 24, 24, 4), dtype="float32") = R.nn.conv2d(
            gv5,
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
        gv7: R.Tensor((2, 4, 24, 24), dtype="float32") = R.transpose(gv6, axes=[0, 3, 1, 2])
        return gv7


def test_conv2d_add_relu_conv2d():
    @I.ir_module
    class Conv2dAddReLUConv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
            gv3: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv2)
            gv4: R.Tensor((2, 4, 24, 24), "float32") = R.nn.conv2d(
                gv3, w, kernel_size=[3, 3], out_dtype="float32"
            )
            return gv4

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dAddReLUConv2d)
    tvm.ir.assert_structural_equal(mod, conv2d_add_relu_conv2d)


@I.ir_module
class conv2d_fma_relu_conv2d:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"),
        w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        scale: R.Tensor((2, 4, 26, 26), dtype="float32"),
        bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.transpose(scale, axes=[0, 2, 3, 1])
        gv4: R.Tensor((2, 26, 26, 4), dtype="float32") = R.transpose(bias, axes=[0, 2, 3, 1])
        gv5: R.Tensor((2, 26, 26, 4), dtype="float32") = R.ewise_fma(gv2, gv3, gv4)
        gv6: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv5)
        gv7: R.Tensor((2, 24, 24, 4), dtype="float32") = R.nn.conv2d(
            gv6,
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
        gv8: R.Tensor((2, 4, 24, 24), dtype="float32") = R.transpose(gv7, axes=[0, 3, 1, 2])
        return gv8


def test_conv2d_fma_relu_conv2d():
    @I.ir_module
    class Conv2dFmaReLUConv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            scale: R.Tensor((2, 4, 26, 26), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.ewise_fma(gv, scale, bias)
            gv3: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv2)
            gv4: R.Tensor((2, 4, 24, 24), "float32") = R.nn.conv2d(
                gv3, w, kernel_size=[3, 3], out_dtype="float32"
            )
            return gv4

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dFmaReLUConv2d)
    tvm.ir.assert_structural_equal(mod, conv2d_fma_relu_conv2d)


@I.ir_module
class conv2d_sum:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=2):
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
        gv3: R.Tensor((2, 4), dtype="float32") = R.sum(gv2, axis=[1, 2], keepdims=False)
        return gv3


def test_conv2d_sum():
    @I.ir_module
    class Conv2dSum:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[2, 3])
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dSum)
    tvm.ir.assert_structural_equal(mod, conv2d_sum)


@I.ir_module
class conv2d_sum_keepdim:
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
        gv3: R.Tensor((2, 1, 1, 4), dtype="float32") = R.sum(gv2, axis=[1, 2], keepdims=True)
        gv4: R.Tensor((2, 4, 1, 1), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_sum_keepdim():
    @I.ir_module
    class Conv2dSumKeepDim:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 1, 1), "float32") = R.sum(gv, axis=[2, 3], keepdims=True)
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dSumKeepDim)
    tvm.ir.assert_structural_equal(mod, conv2d_sum_keepdim)


@I.ir_module
class conv2d_transpose:
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
        gv3: R.Tensor((26, 26, 4, 2), dtype="float32") = R.transpose(gv2, axes=[2, 1, 3, 0])
        return gv3


def test_conv2d_transpose():
    @I.ir_module
    class Conv2dT:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((26, 26, 4, 2), "float32") = R.transpose(gv, axes=[3, 2, 1, 0])
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dT)
    tvm.ir.assert_structural_equal(mod, conv2d_transpose)


@I.ir_module
class conv2d_expand_dims:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=6):
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
        gv3: R.Tensor((2, 1, 26, 1, 26, 4), dtype="float32") = R.expand_dims(gv2, axis=[-3, 1])
        gv4: R.Tensor((2, 1, 4, 1, 26, 26), dtype="float32") = R.transpose(
            gv3, axes=[0, 1, 5, 3, 2, 4]
        )
        return gv4


def test_conv2d_expand_dims():
    @I.ir_module
    class Conv2dExpandDims:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=6):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dExpandDims)
    tvm.ir.assert_structural_equal(mod, conv2d_expand_dims)


@I.ir_module
class conv2d_expand_dims_squeeze:
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
        gv3: R.Tensor((2, 1, 26, 1, 26, 4), dtype="float32") = R.expand_dims(gv2, axis=[-3, 1])
        gv4: R.Tensor((2, 26, 26, 4), dtype="float32") = R.squeeze(gv3, axis=[1, 3])
        gv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv4, axes=[0, 3, 1, 2])
        return gv5


def test_conv2d_expand_dims_squeeze():
    @I.ir_module
    class Conv2dExpandDimsSqueeze:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
            gv3: R.Tensor((2, 4, 26, 26), "float32") = R.squeeze(gv2, axis=[1, 3])
            return gv3

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dExpandDimsSqueeze)
    tvm.ir.assert_structural_equal(mod, conv2d_expand_dims_squeeze)


@I.ir_module
class conv2d_strided_slice:
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
        gv3: R.Tensor((2, 9, 7, 2), dtype="float32") = R.strided_slice(
            gv2,
            begin=[0, 0, 0],
            end=[4, 26, 26],
            strides=[2, 3, 4],
            axes=[3, 1, 2],
            slice_mode="end",
        )
        gv4: R.Tensor((2, 2, 9, 7), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_strided_slice():
    @I.ir_module
    class Conv2dStridedSlice:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                gv, begin=[0, 0, 0], end=[4, 26, 26], strides=[2, 3, 4], axes=[1, 2, 3]
            )
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dStridedSlice)
    tvm.ir.assert_structural_equal(mod, conv2d_strided_slice)


@I.ir_module
class conv2d_cumsum:
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.cumsum(gv2, axis=3)
        gv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv2, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_cumsum():
    @I.ir_module
    class Conv2dCumsum:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.cumsum(gv, axis=1)
            return gv

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dCumsum)
    tvm.ir.assert_structural_equal(mod, conv2d_cumsum)


@I.ir_module
class conv2d_cumsum_default_axis:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=1):
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
        gv4: R.Tensor((5408,), dtype="float32") = R.cumsum(gv3, axis=None)
        return gv4


def test_conv2d_cumsum_default_axis():
    @I.ir_module
    class Conv2dCumsumDefaultAxis:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=1):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2 = R.cumsum(gv)
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dCumsumDefaultAxis)
    tvm.ir.assert_structural_equal(mod, conv2d_cumsum_default_axis)


@I.ir_module
class conv2d_relu_concat:
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv2)
        gv4: R.Tensor((2, 26, 26, 8), dtype="float32") = R.concatenate((gv2, gv3), axis=3)
        gv5: R.Tensor((2, 8, 26, 26), dtype="float32") = R.transpose(gv4, axes=[0, 3, 1, 2])
        return gv5


def test_conv2d_relu_concat():
    @I.ir_module
    class Conv2dReLUConcat:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concatenate((gv, gv2), axis=1)
            return gv3

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dReLUConcat)
    tvm.ir.assert_structural_equal(mod, conv2d_relu_concat)


@I.ir_module
class conv2d_relu_concat_split:
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv2)
        gv4: R.Tensor((2, 26, 26, 8), dtype="float32") = R.concatenate((gv2, gv3), axis=3)
        gv5: R.Tuple(
            R.Tensor((2, 26, 26, 4), dtype="float32"), R.Tensor((2, 26, 26, 4), dtype="float32")
        ) = R.split(gv4, indices_or_sections=2, axis=3)
        gv6: R.Tensor((2, 26, 26, 4), dtype="float32") = gv5[0]
        gv7: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv6, axes=[0, 3, 1, 2])
        return gv7


def test_conv2d_relu_concat_split():
    @I.ir_module
    class Conv2dReLUConcatSplit:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concatenate((gv, gv2), axis=1)
            gv4 = R.split(gv3, indices_or_sections=2, axis=1)
            return gv4[0]

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dReLUConcatSplit)
    tvm.ir.assert_structural_equal(mod, conv2d_relu_concat_split)


@I.ir_module
class conv2d_maxpool2d:
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
        gv3: R.Tensor((2, 13, 13, 4), dtype="float32") = R.nn.max_pool2d(
            gv2,
            pool_size=[2, 2],
            strides=[2, 2],
            dilation=[1, 1],
            padding=[0, 0, 0, 0],
            layout="NHWC",
            out_layout="NHWC",
            ceil_mode=False,
        )
        gv4: R.Tensor((2, 4, 13, 13), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_maxpool2d():
    @I.ir_module
    class Conv2dMaxPool2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2 = R.nn.max_pool2d(
                gv,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0],
                layout="NCHW",
                out_layout="NCHW",
            )
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dMaxPool2d)
    tvm.ir.assert_structural_equal(mod, conv2d_maxpool2d)


@I.ir_module
class conv2d_avgpool2d:
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
        gv3: R.Tensor((2, 13, 13, 4), dtype="float32") = R.nn.adaptive_avg_pool2d(
            gv2, output_size=[13, 13], layout="NHWC"
        )
        gv4: R.Tensor((2, 4, 13, 13), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_avgpool2d():
    @I.ir_module
    class Conv2dAvgPool2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2 = R.nn.adaptive_avg_pool2d(gv, output_size=[13, 13], layout="NCHW")
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dAvgPool2d)
    tvm.ir.assert_structural_equal(mod, conv2d_avgpool2d)


@I.ir_module
class conv2d_softmax:
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.softmax(gv2, axis=3)
        gv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_softmax():
    @I.ir_module
    class Conv2dSoftmax:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2 = R.nn.softmax(gv, axis=1)
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dSoftmax)
    tvm.ir.assert_structural_equal(mod, conv2d_softmax)


@I.ir_module
class conv2d_batchnorm:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"),
        w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        gamma: R.Tensor((4,), dtype="float32"),
        beta: R.Tensor((4,), dtype="float32"),
        moving_mean: R.Tensor((4,), dtype="float32"),
        moving_var: R.Tensor((4,), dtype="float32"),
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
        gv3: R.Tuple(
            R.Tensor((2, 26, 26, 4), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ) = R.nn.batch_norm(
            gv2,
            gamma,
            beta,
            moving_mean,
            moving_var,
            axis=3,
            epsilon=1e-05,
            center=True,
            scale=True,
        )
        gv4: R.Tensor((2, 26, 26, 4), dtype="float32") = gv3[0]
        gv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv4, axes=[0, 3, 1, 2])
        return gv5


def test_conv2d_batchnorm():
    @I.ir_module
    class Conv2dBatchNorm:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            gamma: R.Tensor((4,), dtype="float32"),
            beta: R.Tensor((4,), dtype="float32"),
            moving_mean: R.Tensor((4,), dtype="float32"),
            moving_var: R.Tensor((4,), dtype="float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tuple(
                R.Tensor((2, 4, 26, 26), dtype="float32"),
                R.Tensor((4,), dtype="float32"),
                R.Tensor((4,), dtype="float32"),
            ) = R.nn.batch_norm(gv, gamma, beta, moving_mean, moving_var, axis=1)
            return gv2[0]

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dBatchNorm)
    tvm.ir.assert_structural_equal(mod, conv2d_batchnorm)


@I.ir_module
class conv2d_layernorm:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"),
        w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        gamma: R.Tensor((26, 26), dtype="float32"),
        beta: R.Tensor((26, 26), dtype="float32"),
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
        gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.layer_norm(
            gv2, gamma, beta, axis=[1, 2], epsilon=1e-05, center=True, scale=True
        )
        gv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_layernorm():
    @I.ir_module
    class Conv2dLayerNorm:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            gamma: R.Tensor((26, 26), dtype="float32"),
            beta: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.layer_norm(
                gv, gamma, beta, axis=[-2, -1]
            )
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dLayerNorm)
    tvm.ir.assert_structural_equal(mod, conv2d_layernorm)


@I.ir_module
class conv2d_resize2d:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"),
        w: R.Tensor((4, 3, 3, 3), dtype="float32")
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
        gv3: R.Tensor((2, 52, 52, 4), dtype="float32") = R.image.resize2d(
            gv2,
            size=[52, 52],
            roi=[0.000000, 0.000000, 0.000000, 0.000000],
            layout="NHWC",
            method="linear",
            coordinate_transformation_mode="half_pixel",
            rounding_method="round",
            cubic_alpha=-0.5,
            cubic_exclude=0,
            extrapolation_value=0,
        )
        gv4: R.Tensor((2, 4, 52, 52), dtype="float32") = R.transpose(gv3, axes=[0, 3, 1, 2])
        return gv4


def test_conv2d_resize2d():
    @I.ir_module
    class Conv2dResize2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float32"
            )
            gv2 = R.image.resize2d(gv, size=[52, 52], layout="NCHW")
            return gv2

    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(Conv2dResize2d)
    tvm.ir.assert_structural_equal(mod, conv2d_resize2d)


if __name__ == "__main__":
    test_conv2d()
    test_conv2d_relu()
    test_relu_conv2d_relu()
    test_conv2d_relu_tanh()
    test_conv2d_add()
    test_conv2d_add_relu_conv2d()
    test_conv2d_fma_relu_conv2d()
    test_conv2d_sum()
    test_conv2d_sum_keepdim()
    test_conv2d_transpose()
    test_conv2d_expand_dims()
    test_conv2d_expand_dims_squeeze()
    test_conv2d_strided_slice()
    test_conv2d_cumsum()
    test_conv2d_cumsum_default_axis()
    test_conv2d_relu_concat()
    test_conv2d_relu_concat_split()
    test_conv2d_maxpool2d()
    test_conv2d_avgpool2d()
    test_conv2d_softmax()
    test_conv2d_batchnorm()
    test_conv2d_layernorm()
    test_conv2d_resize2d()
