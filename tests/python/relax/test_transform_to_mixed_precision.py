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


def _assert_test(input, expected):
    mod = ToMixedPrecision()(input)
    tvm.ir.assert_structural_equal(mod, expected)


def test_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
            gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                gv,
                gv1,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            return gv2

    _assert_test(Input, Expected)


def test_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
            gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                gv,
                gv1,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.relu(gv2)
            return gv21

    _assert_test(Input, Expected)


def test_relu_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            x0: R.Tensor((2, 3, 28, 28), "float32") = R.nn.relu(x)
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x0, w, out_dtype="float32")
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
            x0: R.Tensor((2, 3, 28, 28), dtype="float16") = R.nn.relu(gv)
            gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                x0,
                gv1,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.relu(gv2)
            return gv21

    _assert_test(Input, Expected)


def test_conv2d_relu_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            w2: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
            gv3: R.Tensor((2, 4, 24, 24), "float32") = R.nn.conv2d(gv2, w2, out_dtype="float32")
            return gv3

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            w2: R.Tensor((4, 4, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
            gv2: R.Tensor((4, 4, 3, 3), dtype="float16") = R.astype(w2, dtype="float16")
            gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                gv,
                gv1,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            gv31: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(gv3, dtype="float16")
            gv21: R.Tensor((2, 4, 26, 26), dtype="float16") = R.nn.relu(gv31)
            gv32: R.Tensor((2, 4, 24, 24), dtype="float32") = R.nn.conv2d(
                gv21,
                gv2,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            return gv32

    _assert_test(Input, Expected)


def test_gemm_add_silu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 320), "float32"),
            w1: R.Tensor((320, 1280), "float32"),
            w2: R.Tensor((2, 1280), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            gv0: R.Tensor((2, 1280), "float32") = R.matmul(x, w1, out_dtype="float32")
            gv1: R.Tensor((2, 1280), "float32") = R.add(gv0, w2)
            gv2: R.Tensor((2, 1280), "float32") = R.nn.silu(gv1)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 320), dtype="float32"),
            w1: R.Tensor((320, 1280), dtype="float32"),
            w2: R.Tensor((2, 1280), dtype="float32"),
        ) -> R.Tensor((2, 1280), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 320), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tensor((320, 1280), dtype="float16") = R.astype(w1, dtype="float16")
            gv0: R.Tensor((2, 1280), dtype="float32") = R.matmul(gv, gv1, out_dtype="float32")
            gv11: R.Tensor((2, 1280), dtype="float32") = R.add(gv0, w2)
            gv2: R.Tensor((2, 1280), dtype="float32") = R.nn.silu(gv11)
            return gv2

    _assert_test(Input, Expected)


def test_tuple():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            w_2: R.Tensor((4, 4, 3, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
            gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
            gv3 = (gv, gv2)
            gv4 = (gv3, gv2)
            gv5 = gv4[0]
            gv6 = gv5[0]
            gv7 = R.nn.conv2d(gv6, w_2, out_dtype="float32")
            return gv7

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            w_2: R.Tensor((4, 4, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tensor((4, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
            gv2: R.Tensor((4, 4, 3, 3), dtype="float16") = R.astype(w_2, dtype="float16")
            gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                gv,
                gv1,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.conv2d(
                gv,
                gv1,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            gv31: R.Tuple(
                R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")
            ) = (gv3, gv21)
            gv4: R.Tuple(
                R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                ),
                R.Tensor((2, 4, 26, 26), dtype="float32"),
            ) = (gv31, gv21)
            gv5: R.Tuple(
                R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")
            ) = gv4[0]
            gv6: R.Tensor((2, 4, 26, 26), dtype="float32") = gv5[0]
            gv32: R.Tensor((2, 4, 26, 26), dtype="float16") = R.astype(gv6, dtype="float16")
            gv7: R.Tensor((2, 4, 24, 24), dtype="float32") = R.nn.conv2d(
                gv32,
                gv2,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            return gv7

    _assert_test(Input, Expected)


def test_concat_matmul():
    @I.ir_module
    class Input:
        @R.function
        def main(
            lv10: R.Tensor((2, 160), "float32"),
            lv12: R.Tensor((2, 160), "float32"),
            w: R.Tensor((320, 1280), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            lv13: R.Tensor((2, 320), "float32") = R.concat((lv10, lv12), axis=-1)
            lv14: R.Tensor((2, 1280), "float32") = R.matmul(lv13, w, out_dtype="float32")
            return lv14

    @I.ir_module
    class Expected:
        @R.function
        def main(
            lv10: R.Tensor((2, 160), dtype="float32"),
            lv12: R.Tensor((2, 160), dtype="float32"),
            w: R.Tensor((320, 1280), dtype="float32"),
        ) -> R.Tensor((2, 1280), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 160), dtype="float16") = R.astype(lv10, dtype="float16")
            gv1: R.Tensor((2, 160), dtype="float16") = R.astype(lv12, dtype="float16")
            gv2: R.Tensor((320, 1280), dtype="float16") = R.astype(w, dtype="float16")
            lv13: R.Tensor((2, 320), dtype="float16") = R.concat((gv, gv1), axis=-1)
            lv14: R.Tensor((2, 1280), dtype="float32") = R.matmul(lv13, gv2, out_dtype="float32")
            return lv14

    _assert_test(Input, Expected)


def test_conv2d_softmax():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((3, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 3, 26, 26), "float32") = R.nn.conv2d(x, w, padding=(1, 1))
            gv1: R.Tensor((2, 3, 26, 26), "float32") = R.nn.softmax(x, axis=1)
            gv2 = R.add(gv, gv1)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((3, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 3, 26, 26), dtype="float32"):
            # block 0
            gv: R.Tensor((2, 3, 28, 28), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tensor((3, 3, 3, 3), dtype="float16") = R.astype(w, dtype="float16")
            gv2: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.conv2d(
                gv,
                gv1,
                strides=[1, 1],
                padding=[1, 1, 1, 1],
                dilation=[1, 1],
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="float32",
            )
            gv21: R.Tensor((2, 3, 28, 28), dtype="float32") = R.astype(gv, dtype="float32")
            gv11: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.softmax(gv21, axis=1)
            gv22: R.Tensor((2, 3, 28, 28), dtype="float32") = R.add(gv2, gv11)
            return gv22

    _assert_test(Input, Expected)


if __name__ == "__main__":
    test_conv2d()
    test_conv2d_relu()
    test_relu_conv2d_relu()
    test_conv2d_relu_conv2d()
    test_gemm_add_silu()
    test_tuple()
    test_concat_matmul()
    test_conv2d_softmax()
