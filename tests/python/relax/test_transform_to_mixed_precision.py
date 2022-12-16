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
from tvm.relax.transform import ToMixedPrecision
from tvm.script.parser import ir as I, relax as R
from tvm.relax.transform import mixed_precision


@I.ir_module
class conv2d:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        # block 0
        gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.cast(x, dtype="float16")
        gv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.cast(w, dtype="float16")
        gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
            gv,
            gv1,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=None,
            kernel_size=[3, 3],
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_layout="",
            out_dtype="float32",
        )
        gv3: R.Tensor((2, 4, 26, 26), dtype="float16") = R.cast(gv2, dtype="float16")
        return gv2


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

    mod = ToMixedPrecision()(Conv2d)
    tvm.ir.assert_structural_equal(mod, conv2d)


@I.ir_module
class conv2d_relu:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        # block 0
        gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.cast(x, dtype="float16")
        gv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.cast(w, dtype="float16")
        gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
            gv,
            gv1,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=None,
            kernel_size=[3, 3],
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_layout="",
            out_dtype="float32",
        )
        gv3: R.Tensor((2, 4, 26, 26), dtype="float16") = R.cast(gv2, dtype="float16")
        gv4: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(gv3)
        gv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.cast(gv4, dtype="float32")
        return gv5


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

    mod = ToMixedPrecision()(Conv2dReLU)
    tvm.ir.assert_structural_equal(mod, conv2d_relu)


@tvm.script.ir_module
class relu_conv2d_relu:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        # block 0
        gv: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.relu(x)
        gv1: R.Tensor((2, 3, 28, 28), dtype="float16") = R.cast(gv, dtype="float16")
        gv2: R.Tensor((4, 3, 3, 3), dtype="float16") = R.cast(w, dtype="float16")
        gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
            gv1,
            gv2,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=None,
            kernel_size=[3, 3],
            data_layout="NCHW",
            kernel_layout="OIHW",
            out_layout="",
            out_dtype="float32",
        )
        gv4: R.Tensor((2, 4, 26, 26), dtype="float16") = R.cast(gv3, dtype="float16")
        gv5: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(gv4)
        gv6: R.Tensor((2, 4, 26, 26), dtype="float32") = R.cast(gv5, dtype="float32")
        return gv6


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

    mod = ToMixedPrecision()(ReLUConv2dReLU)
    tvm.ir.assert_structural_equal(mod, relu_conv2d_relu)


@tvm.script.ir_module
class gemm_add_silu:
    @R.function
    def main(
        lv13: R.Tensor((2, 320), dtype="float32"),
        w1: R.Tensor((320, 1280), dtype="float32"),
        w2: R.Tensor((2, 1280), dtype="float32"),
    ) -> R.Tensor(None, dtype="float32", ndim=2):
        # block 0
        gv: R.Tensor((2, 320), dtype="float16") = R.cast(lv13, dtype="float16")
        gv1: R.Tensor((320, 1280), dtype="float16") = R.cast(w1, dtype="float16")
        gv2: R.Tensor((2, 1280), dtype="float32") = R.nn.matmul(gv, gv1, out_dtype="float32")
        gv3: R.Tensor((2, 1280), dtype="float16") = R.cast(gv2, dtype="float16")
        gv4: R.Tensor((2, 1280), dtype="float32") = R.add(gv2, w2)
        gv5: R.Tensor((2, 1280), dtype="float32") = R.nn.silu(gv4)
        return gv5


def test_gemm_add_silu():
    @I.ir_module
    class GemmAddSiLU:
        @R.function
        def main(
            lv13: R.Tensor((2, 320), "float32"),
            w1: R.Tensor((320, 1280), "float32"),
            w2: R.Tensor((2, 1280), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            lv14: R.Tensor((2, 1280), "float32") = relax.nn.matmul(lv13, w1, out_dtype="float32")
            lv15: R.Tensor((2, 1280), "float32") = R.add(lv14, w2)
            lv16: R.Tensor((2, 1280), "float32") = relax.nn.silu(lv15)
            return lv16

    mod = ToMixedPrecision()(GemmAddSiLU)
    tvm.ir.assert_structural_equal(mod, gemm_add_silu)


@tvm.script.ir_module
class concat:
    @R.function
    def main(lv5: R.Tensor((2, 160), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
        # block 0
        gv: R.Tensor((2, 160), dtype="float32") = R.sin(lv5)
        gv1: R.Tensor((2, 160), dtype="float32") = R.cos(lv5)
        gv2: R.Tensor((2, 320), dtype="float32") = R.concatenate((gv, gv1), axis=-1)
        return gv2


def test_concat():
    @I.ir_module
    class Concat:
        @R.function
        def main(lv5: R.Tensor((2, 160), "float32")) -> R.Tensor(None, "float32", ndim=2):
            lv6: R.Tensor((2, 160), "float32") = R.sin(lv5)
            lv7: R.Tensor((2, 160), "float32") = R.cos(lv5)
            lv8: R.Tensor((2, 320), "float32") = R.concatenate((lv6, lv7), axis=-1)
            return lv8

    mod = ToMixedPrecision()(Concat)
    tvm.ir.assert_structural_equal(mod, concat)


@tvm.script.ir_module
class concat_matmul:
    @R.function
    def main(
        lv10: R.Tensor((2, 160), dtype="float32"),
        lv12: R.Tensor((2, 160), dtype="float32"),
        w: R.Tensor((320, 1280), dtype="float32"),
    ) -> R.Tensor(None, dtype="float32", ndim=2):
        # block 0
        gv: R.Tensor((2, 320), dtype="float32") = R.concatenate((lv10, lv12), axis=-1)
        gv1: R.Tensor((2, 320), dtype="float16") = R.cast(gv, dtype="float16")
        gv2: R.Tensor((320, 1280), dtype="float16") = R.cast(w, dtype="float16")
        gv3: R.Tensor((2, 1280), dtype="float32") = R.nn.matmul(gv1, gv2, out_dtype="float32")
        gv4: R.Tensor((2, 1280), dtype="float16") = R.cast(gv3, dtype="float16")
        return gv3


def test_concat_matmul():
    @I.ir_module
    class ConcatMatmul:
        @R.function
        def main(
            lv10: R.Tensor((2, 160), "float32"),
            lv12: R.Tensor((2, 160), "float32"),
            w: R.Tensor((320, 1280), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            lv13: R.Tensor((2, 320), "float32") = R.concatenate((lv10, lv12), axis=-1)
            lv14: R.Tensor((2, 1280), "float32") = relax.nn.matmul(lv13, w, out_dtype="float32")
            return lv14

    mod = ToMixedPrecision()(ConcatMatmul)
    tvm.ir.assert_structural_equal(mod, concat_matmul)


if __name__ == "__main__":
    test_conv2d()
    test_conv2d_relu()
    test_relu_conv2d_relu()
    test_gemm_add_silu()
    test_concat()
    test_concat_matmul()
