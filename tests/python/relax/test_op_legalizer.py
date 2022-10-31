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
from tvm.error import DiagnosticError
from tvm.relax.transform import OperatorLegalizer
from tvm.script._parser import ir as I, relax as R, tir as T
import tvm.testing


def test_conv2d():
    @I.ir_module
    class Conv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.conv2d(x, w, kernel_size=[3, 3])
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(conv2d, (x, w), (2, 4, 26, 26), dtype="float32")
            return gv

        @T.prim_func
        def conv2d(
            rxplaceholder: T.Buffer[(2, 3, 28, 28), "float32"],
            rxplaceholder_1: T.Buffer[(4, 3, 3, 3), "float32"],
            conv2d_nchw: T.Buffer[(2, 4, 26, 26), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "conv2d", "tir.noalias": True})
            pad_temp = T.alloc_buffer([2, 3, 28, 28], dtype="float32")
            for i0, i1, i2, i3 in T.grid(2, 3, 28, 28):
                with T.block("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(2, 4, 26, 26, 3, 3, 3):
                with T.block("conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = T.axis.remap(
                        "SSSSRRR", [i0, i1, i2, i3, i4, i5, i6]
                    )
                    T.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
                    T.writes(conv2d_nchw[nn, ff, yy, xx])
                    with T.init():
                        conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
                    conv2d_nchw[nn, ff, yy, xx] = (
                        conv2d_nchw[nn, ff, yy, xx]
                        + pad_temp[nn, rc, yy + ry, xx + rx] * rxplaceholder_1[ff, rc, ry, rx]
                    )

    mod = OperatorLegalizer(Conv2d).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_add():
    @I.ir_module
    class Add:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.add(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(add, (x, y), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def add(
            rxplaceholder: T.Buffer[(2, 3), "float32"],
            rxplaceholder_1: T.Buffer[(2, 3), "float32"],
            T_add: T.Buffer[(2, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "add", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = rxplaceholder[ax0, ax1] + rxplaceholder_1[ax0, ax1]

    mod = OperatorLegalizer(Add).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_subtract():
    @I.ir_module
    class Subtract:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.subtract(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(subtract, (x, y), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def subtract(
            rxplaceholder: T.Buffer[(2, 3), "float32"],
            rxplaceholder_1: T.Buffer[(2, 3), "float32"],
            T_subtract: T.Buffer[(2, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "subtract", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("T_subtract"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_subtract[ax0, ax1])
                    T_subtract[ax0, ax1] = rxplaceholder[ax0, ax1] - rxplaceholder_1[ax0, ax1]

    mod = OperatorLegalizer(Subtract).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_multiply():
    @I.ir_module
    class Multiply:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.multiply(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(multiply, (x, y), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def multiply(
            rxplaceholder: T.Buffer[(2, 3), "float32"],
            rxplaceholder_1: T.Buffer[(2, 3), "float32"],
            T_multiply: T.Buffer[(2, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "multiply", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * rxplaceholder_1[ax0, ax1]

    mod = OperatorLegalizer(Multiply).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_divide():
    @I.ir_module
    class Divide:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.divide(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(divide, (x, y), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def divide(
            rxplaceholder: T.Buffer[(2, 3), "float32"],
            rxplaceholder_1: T.Buffer[(2, 1), "float32"],
            T_divide: T.Buffer[(2, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "divide", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, 0])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder[ax0, ax1] / rxplaceholder_1[ax0, 0]

    mod = OperatorLegalizer(Divide).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_floor_divide():
    @I.ir_module
    class FloorDivide:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.floor_divide(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(floor_divide, (x, y), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def floor_divide(
            rxplaceholder: T.Buffer[(2, 3), "float32"],
            rxplaceholder_1: T.Buffer[(2, 1), "float32"],
            T_floor_divide: T.Buffer[(2, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "floor_divide", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("T_floor_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, 0])
                    T.writes(T_floor_divide[ax0, ax1])
                    T_floor_divide[ax0, ax1] = T.floor(
                        rxplaceholder[ax0, ax1] / rxplaceholder_1[ax0, 0], dtype="float32"
                    )

    mod = OperatorLegalizer(FloorDivide).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sin():
    @I.ir_module
    class Sin:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.sin(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sin, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def sin(
            rxplaceholder: T.Buffer[(2, 3), "float32"], compute: T.Buffer[(2, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "sin", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sin(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Sin).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cos():
    @I.ir_module
    class Cos:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.cos(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(cos, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def cos(
            rxplaceholder: T.Buffer[(2, 3), "float32"], compute: T.Buffer[(2, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "cos", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.cos(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Cos).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sqrt():
    @I.ir_module
    class Sqrt:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.sqrt(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sqrt, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def sqrt(
            rxplaceholder: T.Buffer[(2, 3), "float32"], compute: T.Buffer[(2, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "sqrt", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sqrt(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Sqrt).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_relu():
    @I.ir_module
    class Relu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.relu(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(relu, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def relu(
            rxplaceholder: T.Buffer[(2, 3), "float32"], compute: T.Buffer[(2, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "relu", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.max(rxplaceholder[i0_1, i1_1], T.float32(0))

    mod = OperatorLegalizer(Relu).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_gelu():
    @I.ir_module
    class Gelu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.gelu(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(gelu, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def gelu(
            rxplaceholder: T.Buffer[(2, 3), "float32"], compute: T.Buffer[(2, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "gelu", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = (
                        T.float32(0.5)
                        * rxplaceholder[i0_1, i1_1]
                        * (
                            T.float32(1)
                            + T.tanh(
                                T.float32(0.79788456080286541)
                                * (
                                    rxplaceholder[i0_1, i1_1]
                                    + T.float32(0.044714999999999998)
                                    * T.power(
                                        rxplaceholder[i0_1, i1_1], T.float32(3), dtype="float32"
                                    )
                                ),
                                dtype="float32",
                            )
                        )
                    )

    mod = OperatorLegalizer(Gelu).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_silu():
    @I.ir_module
    class Silu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.silu(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sigmoid, (x,), (2, 3), dtype="float32")
            gv1 = R.call_tir(multiply, (x, gv), (2, 3), dtype="float32")
            return gv1

        @T.prim_func
        def sigmoid(
            rxplaceholder: T.Buffer[(2, 3), "float32"], compute: T.Buffer[(2, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "sigmoid", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1], dtype="float32")

        @T.prim_func
        def multiply(
            rxplaceholder: T.Buffer[(2, 3), "float32"],
            rxplaceholder_1: T.Buffer[(2, 3), "float32"],
            T_multiply: T.Buffer[(2, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "multiply", "tir.noalias": True})
            for i0, i1 in T.grid(2, 3):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * rxplaceholder_1[ax0, ax1]

    mod = OperatorLegalizer(Silu).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_reshape():
    @I.ir_module
    class Reshape:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((8, 3), "float32") = R.reshape(x, (8, 3))
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(reshape, (x,), (8, 3), dtype="float32")
            return gv

        @T.prim_func
        def reshape(
            rxplaceholder: T.Buffer[(1, 2, 3, 4), "float32"], T_reshape: T.Buffer[(8, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "reshape", "tir.noalias": True})
            for i0, i1 in T.grid(8, 3):
                with T.block("T_reshape"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(
                        rxplaceholder[
                            0,
                            (ax0 * 3 + ax1) % 24 // 12,
                            (ax0 * 3 + ax1) % 12 // 4,
                            (ax0 * 3 + ax1) % 4,
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        0,
                        (ax0 * 3 + ax1) % 24 // 12,
                        (ax0 * 3 + ax1) % 12 // 4,
                        (ax0 * 3 + ax1) % 4,
                    ]

    mod = OperatorLegalizer(Reshape).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_reshape_dim_inference():
    @I.ir_module
    class Reshape:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
            gv: R.Tensor((8, 1, 3), "float32") = R.reshape(x, (8, -1, 3))
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
            gv = R.call_tir(reshape, (x,), (8, 1, 3), dtype="float32")
            return gv

        @T.prim_func
        def reshape(
            rxplaceholder: T.Buffer[(1, 2, 3, 4), "float32"],
            T_reshape: T.Buffer[(8, 1, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "reshape", "tir.noalias": True})
            for i0, i1, i2 in T.grid(8, 1, 3):
                with T.block("T_reshape"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(
                        rxplaceholder[
                            0,
                            (ax0 * 3 + ax1 * 3 + ax2) % 24 // 12,
                            (ax0 * 3 + ax1 * 3 + ax2) % 12 // 4,
                            (ax0 * 3 + ax1 * 3 + ax2) % 4,
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1, ax2])
                    T_reshape[ax0, ax1, ax2] = rxplaceholder[
                        0,
                        (ax0 * 3 + ax1 * 3 + ax2) % 24 // 12,
                        (ax0 * 3 + ax1 * 3 + ax2) % 12 // 4,
                        (ax0 * 3 + ax1 * 3 + ax2) % 4,
                    ]

    mod = OperatorLegalizer(Reshape).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_transpose():
    @I.ir_module
    class Transpose:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 3, 1), "float32") = R.transpose(x, axes=[1, -1, 2, -4])
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(transpose, (x,), (2, 4, 3, 1), dtype="float32")
            return gv

        @T.prim_func
        def transpose(
            rxplaceholder: T.Buffer[(1, 2, 3, 4), "float32"],
            T_transpose: T.Buffer[(2, 4, 3, 1), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "transpose", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(2, 4, 3, 1):
                with T.block("T_transpose"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax3, ax0, ax2, ax1])
                    T.writes(T_transpose[ax0, ax1, ax2, ax3])
                    T_transpose[ax0, ax1, ax2, ax3] = rxplaceholder[ax3, ax0, ax2, ax1]

    mod = OperatorLegalizer(Transpose).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_concatenate():
    @I.ir_module
    class Concatenate:
        @R.function
        def main(
            x1: R.Tensor((1, 2, 3), "float32"),
            x2: R.Tensor((1, 3, 3), "float32"),
            x3: R.Tensor((1, 4, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=3):
            gv: R.Tensor((1, 9, 3), "float32") = R.concatenate((x1, x2, x3), axis=1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x1: R.Tensor((1, 2, 3), "float32"),
            x2: R.Tensor((1, 3, 3), "float32"),
            x3: R.Tensor((1, 4, 3), "float32"),
        ) -> R.Tensor(None, "float32", ndim=3):
            gv = R.call_tir(concatenate, (x1, x2, x3), (1, 9, 3), dtype="float32")
            return gv

        @T.prim_func
        def concatenate(
            rxplaceholder: T.Buffer[(1, 2, 3), "float32"],
            rxplaceholder_1: T.Buffer[(1, 3, 3), "float32"],
            rxplaceholder_2: T.Buffer[(1, 4, 3), "float32"],
            T_concat: T.Buffer[(1, 9, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "concatenate", "tir.noalias": True})
            for i0, i1, i2 in T.grid(1, 9, 3):
                with T.block("T_concat"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(
                        rxplaceholder_2[ax0, ax1 - 5, ax2],
                        rxplaceholder_1[ax0, ax1 - 2, ax2],
                        rxplaceholder[ax0, ax1, ax2],
                    )
                    T.writes(T_concat[ax0, ax1, ax2])
                    T_concat[ax0, ax1, ax2] = T.if_then_else(
                        5 <= ax1,
                        rxplaceholder_2[ax0, ax1 - 5, ax2],
                        T.if_then_else(
                            2 <= ax1,
                            rxplaceholder_1[ax0, ax1 - 2, ax2],
                            rxplaceholder[ax0, ax1, ax2],
                            dtype="float32",
                        ),
                        dtype="float32",
                    )

    mod = OperatorLegalizer(Concatenate).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cumsum():
    @I.ir_module
    class Cumsum:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
            gv: R.Tensor((2, 3, 4), "float32") = R.cumsum(x, axis=-2)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
            gv = R.call_tir(cumsum, (x,), (2, 3, 4), dtype="float32")
            return gv

        @T.prim_func
        def cumsum(
            rxplaceholder: T.Buffer[(2, 3, 4), "float32"], out_buf: T.Buffer[(2, 3, 4), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "cumsum", "tir.noalias": True})
            with T.block("cumsum_generic"):
                T.reads(rxplaceholder[0:2, 0:3, 0:4])
                T.writes(out_buf[0:2, 0:3, 0:4])
                for fused in T.parallel(8):
                    out_buf[
                        (fused // 4 * 3 * 4 + fused % 4) // 4 // 3,
                        (fused // 4 * 3 * 4 + fused % 4) // 4 % 3,
                        (fused // 4 * 3 * 4 + fused % 4) % 4,
                    ] = rxplaceholder[
                        (fused // 4 * 3 * 4 + fused % 4) // 4 // 3,
                        (fused // 4 * 3 * 4 + fused % 4) // 4 % 3,
                        (fused // 4 * 3 * 4 + fused % 4) % 4,
                    ]
                    for v_k in T.serial(2):
                        out_buf[
                            (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1) * 4) // 4 // 3,
                            (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1) * 4) // 4 % 3,
                            (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1) * 4) % 4,
                        ] = (
                            out_buf[
                                (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1 - 1) * 4) // 4 // 3,
                                (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1 - 1) * 4) // 4 % 3,
                                (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1 - 1) * 4) % 4,
                            ]
                            + rxplaceholder[
                                (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1) * 4) // 4 // 3,
                                (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1) * 4) // 4 % 3,
                                (fused // 4 * 3 * 4 + fused % 4 + (v_k + 1) * 4) % 4,
                            ]
                        )

    mod = OperatorLegalizer(Cumsum).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cumsum_without_specified_axis():
    @I.ir_module
    class Cumsum:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=1):
            gv: R.Tensor((24,), "float32") = R.cumsum(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=1):
            gv = R.call_tir(cumsum, (x,), (24,), dtype="float32")
            return gv

        @T.prim_func
        def cumsum(
            rxplaceholder: T.Buffer[(2, 3, 4), "float32"], out_buf: T.Buffer[24, "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "cumsum", "tir.noalias": True})
            with T.block("cumsum_generic"):
                T.reads(rxplaceholder[0:2, 0:3, 0:4])
                T.writes(out_buf[0:24])
                for fused in T.parallel(1):
                    out_buf[fused * 24] = rxplaceholder[
                        fused * 24 // 4 // 3, fused * 24 // 4 % 3, fused * 24 % 4
                    ]
                    for v_k in T.serial(23):
                        out_buf[fused * 24 + (v_k + 1)] = (
                            out_buf[fused * 24 + (v_k + 1 - 1)]
                            + rxplaceholder[
                                (fused * 24 + (v_k + 1)) // 4 // 3,
                                (fused * 24 + (v_k + 1)) // 4 % 3,
                                (fused * 24 + (v_k + 1)) % 4,
                            ]
                        )

    mod = OperatorLegalizer(Cumsum).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_expand_dims():
    @I.ir_module
    class ExpandDims:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=8):
            gv: R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32") = R.expand_dims(
                x, axis=[-1, 1, -6, 3, 5]
            )
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=8):
            gv = R.call_tir(expand_dims, (x,), (2, 1, 1, 1, 3, 1, 4, 1), dtype="float32")
            return gv

        @T.prim_func
        def expand_dims(
            rxplaceholder: T.Buffer[(2, 3, 4), "float32"],
            compute: T.Buffer[(2, 1, 1, 1, 3, 1, 4, 1), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "expand_dims", "tir.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(2, 1, 1, 1, 3, 1, 4, 1):
                with T.block("compute"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1 = T.axis.remap(
                        "SSSSSSSS", [i0, i1, i2, i3, i4, i5, i6, i7]
                    )
                    T.reads(rxplaceholder[i0_1, i4_1, i6_1])
                    T.writes(compute[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1])
                    compute[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1] = rxplaceholder[
                        i0_1, i4_1, i6_1
                    ]

    mod = OperatorLegalizer(ExpandDims).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_trilu():
    @I.ir_module
    class Trilu:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
            gv: R.Tensor((2, 3, 4), "float32") = R.trilu(x, k=0, is_upper=False)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=3):
            gv = R.call_tir(trilu, (x,), (2, 3, 4), dtype="float32")
            return gv

        @T.prim_func
        def trilu(
            rxplaceholder: T.Buffer[(2, 3, 4), "float32"], trilu: T.Buffer[(2, 3, 4), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "trilu", "tir.noalias": True})
            for i0, i1, i2 in T.grid(2, 3, 4):
                with T.block("trilu"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(
                        i2_1 <= i1_1, rxplaceholder[i0_1, i1_1, i2_1], T.float32(0)
                    )

    mod = OperatorLegalizer(Trilu).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cast():
    @I.ir_module
    class Cast:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "int32", ndim=3):
            gv: R.Tensor((2, 3, 4), "int32") = R.cast(x, "int32")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor(None, "int32", ndim=3):
            gv = R.call_tir(cast, (x,), (2, 3, 4), dtype="int32")
            return gv

        @T.prim_func
        def cast(
            rxplaceholder: T.Buffer[(2, 3, 4), "float32"], compute: T.Buffer[(2, 3, 4), "int32"]
        ) -> None:
            T.func_attr({"global_symbol": "cast", "tir.noalias": True})
            for i0, i1, i2 in T.grid(2, 3, 4):
                with T.block("compute"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(compute[i0_1, i1_1, i2_1])
                    compute[i0_1, i1_1, i2_1] = T.cast(rxplaceholder[i0_1, i1_1, i2_1], "int32")

    mod = OperatorLegalizer(Cast).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layer_norm():
    @I.ir_module
    class LayerNorm:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"),
            gamma: R.Tensor((4, 5), "float32"),
            beta: R.Tensor((4, 5), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.layer_norm(x, gamma, beta, axis=[-2, -1])
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"),
            gamma: R.Tensor((4, 5), "float32"),
            beta: R.Tensor((4, 5), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(layer_norm, (x, gamma, beta), (2, 3, 4, 5), dtype="float32")
            return gv

        @T.prim_func
        def layer_norm(
            rxplaceholder: T.Buffer[(2, 3, 4, 5), "float32"],
            rxplaceholder_1: T.Buffer[(4, 5), "float32"],
            rxplaceholder_2: T.Buffer[(4, 5), "float32"],
            T_add: T.Buffer[(2, 3, 4, 5), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "layer_norm", "tir.noalias": True})
            rxplaceholder_red = T.alloc_buffer([2, 3, 1, 1], dtype="float32")
            T_divide = T.alloc_buffer([2, 3, 1, 1], dtype="float32")
            T_subtract = T.alloc_buffer([2, 3, 4, 5], dtype="float32")
            T_subtract_1 = T.alloc_buffer([2, 3, 4, 5], dtype="float32")
            T_subtract_2 = T.alloc_buffer([2, 3, 4, 5], dtype="float32")
            T_multiply = T.alloc_buffer([2, 3, 4, 5], dtype="float32")
            T_multiply_red = T.alloc_buffer([2, 3, 1, 1], dtype="float32")
            T_divide_1 = T.alloc_buffer([2, 3, 1, 1], dtype="float32")
            T_add_1 = T.alloc_buffer([2, 3, 1, 1], dtype="float32")
            compute = T.alloc_buffer([2, 3, 1, 1], dtype="float32")
            T_divide_2 = T.alloc_buffer([2, 3, 4, 5], dtype="float32")
            T_multiply_1 = T.alloc_buffer([2, 3, 4, 5], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(2, 3, 1, 1, 4, 5):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k2, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[ax0, ax1, k2, k3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = (
                        rxplaceholder_red[ax0, ax1, ax2, ax3] + rxplaceholder[ax0, ax1, k2, k3]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 1, 1):
                with T.block("T_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = rxplaceholder_red[
                        ax0, ax1, ax2, ax3
                    ] * T.float32(0.050000000000000003)
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide[ax0, ax1, 0, 0])
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = (
                        rxplaceholder[ax0, ax1, ax2, ax3] - T_divide[ax0, ax1, 0, 0]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("T_subtract_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide[ax0, ax1, 0, 0])
                    T.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                    T_subtract_1[ax0, ax1, ax2, ax3] = (
                        rxplaceholder[ax0, ax1, ax2, ax3] - T_divide[ax0, ax1, 0, 0]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("T_subtract_2"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide[ax0, ax1, 0, 0])
                    T.writes(T_subtract_2[ax0, ax1, ax2, ax3])
                    T_subtract_2[ax0, ax1, ax2, ax3] = (
                        rxplaceholder[ax0, ax1, ax2, ax3] - T_divide[ax0, ax1, 0, 0]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_subtract_1[ax0, ax1, ax2, ax3], T_subtract_2[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = (
                        T_subtract_1[ax0, ax1, ax2, ax3] * T_subtract_2[ax0, ax1, ax2, ax3]
                    )
            for i0, i1, i2, i3, i4, i5 in T.grid(2, 3, 1, 1, 4, 5):
                with T.block("T_multiply_red"):
                    ax0, ax1, ax2, ax3, k2, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(T_multiply[ax0, ax1, k2, k3])
                    T.writes(T_multiply_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        T_multiply_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    T_multiply_red[ax0, ax1, ax2, ax3] = (
                        T_multiply_red[ax0, ax1, ax2, ax3] + T_multiply[ax0, ax1, k2, k3]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 1, 1):
                with T.block("T_divide_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_multiply_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide_1[ax0, ax1, ax2, ax3])
                    T_divide_1[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] * T.float32(
                        0.050000000000000003
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 1, 1):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_divide_1[ax0, ax1, ax2, ax3])
                    T.writes(T_add_1[ax0, ax1, ax2, ax3])
                    T_add_1[ax0, ax1, ax2, ax3] = T_divide_1[ax0, ax1, ax2, ax3] + T.float32(
                        1.0000000000000001e-05
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 1, 1):
                with T.block("compute"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_add_1[i0_1, i1_1, i2_1, i3_1])
                    T.writes(compute[i0_1, i1_1, i2_1, i3_1])
                    compute[i0_1, i1_1, i2_1, i3_1] = T.sqrt(
                        T_add_1[i0_1, i1_1, i2_1, i3_1], dtype="float32"
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("T_divide_2"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_subtract[ax0, ax1, ax2, ax3], compute[ax0, ax1, 0, 0])
                    T.writes(T_divide_2[ax0, ax1, ax2, ax3])
                    T_divide_2[ax0, ax1, ax2, ax3] = (
                        T_subtract[ax0, ax1, ax2, ax3] / compute[ax0, ax1, 0, 0]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("T_multiply_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax2, ax3], T_divide_2[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply_1[ax0, ax1, ax2, ax3])
                    T_multiply_1[ax0, ax1, ax2, ax3] = (
                        rxplaceholder_1[ax2, ax3] * T_divide_2[ax0, ax1, ax2, ax3]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("T_add_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_multiply_1[ax0, ax1, ax2, ax3], rxplaceholder_2[ax2, ax3])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = (
                        T_multiply_1[ax0, ax1, ax2, ax3] + rxplaceholder_2[ax2, ax3]
                    )

    mod = OperatorLegalizer(LayerNorm).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_1_4():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            x: R.Tensor((4,), "float32"), y: R.Tensor((2, 3, 4, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            gv: R.Tensor((2, 3, 5), "float32") = R.matmul(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4,), "float32"), y: R.Tensor((2, 3, 4, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            gv = R.call_tir(matmul, (x, y), (2, 3, 5), dtype="float32")
            return gv

        @T.prim_func
        def matmul(
            rxplaceholder: T.Buffer[4, "float32"],
            rxplaceholder_1: T.Buffer[(2, 3, 4, 5), "float32"],
            matmul: T.Buffer[(2, 3, 5), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(2, 3, 5, 4):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[k], rxplaceholder_1[i0_1, i1_1, k, i2_1])
                    T.writes(matmul[i0_1, i1_1, i2_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1] = (
                        matmul[i0_1, i1_1, i2_1]
                        + rxplaceholder[k] * rxplaceholder_1[i0_1, i1_1, k, i2_1]
                    )

    mod = OperatorLegalizer(Matmul).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_1():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((5,), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            gv: R.Tensor((2, 3, 4), "float32") = R.matmul(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((5,), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            gv = R.call_tir(matmul, (x, y), (2, 3, 4), dtype="float32")
            return gv

        @T.prim_func
        def matmul(
            rxplaceholder: T.Buffer[(2, 3, 4, 5), "float32"],
            rxplaceholder_1: T.Buffer[5, "float32"],
            matmul: T.Buffer[(2, 3, 4), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(2, 3, 4, 5):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, k], rxplaceholder_1[k])
                    T.writes(matmul[i0_1, i1_1, i2_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1] = (
                        matmul[i0_1, i1_1, i2_1]
                        + rxplaceholder[i0_1, i1_1, i2_1, k] * rxplaceholder_1[k]
                    )

    mod = OperatorLegalizer(Matmul).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_1_1():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")
        ) -> R.Tensor(None, "float32", ndim=0):
            gv: R.Tensor((), "float32") = R.matmul(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")
        ) -> R.Tensor(None, "float32", ndim=0):
            gv = R.call_tir(matmul, (x, y), (), dtype="float32")
            return gv

        @T.prim_func
        def matmul(
            rxplaceholder: T.Buffer[4, "float32"],
            rxplaceholder_1: T.Buffer[4, "float32"],
            matmul: T.Buffer[(), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0 in T.serial(4):
                with T.block("matmul"):
                    k = T.axis.reduce(4, i0)
                    T.reads(rxplaceholder[k], rxplaceholder_1[k])
                    T.writes(matmul[()])
                    with T.init():
                        matmul[()] = T.float32(0)
                    matmul[()] = matmul[()] + rxplaceholder[k] * rxplaceholder_1[k]

    mod = OperatorLegalizer(Matmul).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_5():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((6, 2, 3, 5, 7), "float32")
        ) -> R.Tensor(None, "float32", ndim=5):
            gv: R.Tensor((6, 2, 3, 4, 7), "float32") = R.matmul(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((6, 2, 3, 5, 7), "float32")
        ) -> R.Tensor(None, "float32", ndim=5):
            gv = R.call_tir(matmul, (x, y), (6, 2, 3, 4, 7), dtype="float32")
            return gv

        @T.prim_func
        def matmul(
            rxplaceholder: T.Buffer[(2, 3, 4, 5), "float32"],
            rxplaceholder_1: T.Buffer[(6, 2, 3, 5, 7), "float32"],
            matmul: T.Buffer[(6, 2, 3, 4, 7), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0, i1, i2, i3, i4, i5 in T.grid(6, 2, 3, 4, 7, 5):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, k = T.axis.remap(
                        "SSSSSR", [i0, i1, i2, i3, i4, i5]
                    )
                    T.reads(
                        rxplaceholder[i1_1, i2_1, i3_1, k],
                        rxplaceholder_1[i0_1, i1_1, i2_1, k, i4_1],
                    )
                    T.writes(matmul[i0_1, i1_1, i2_1, i3_1, i4_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = (
                        matmul[i0_1, i1_1, i2_1, i3_1, i4_1]
                        + rxplaceholder[i1_1, i2_1, i3_1, k]
                        * rxplaceholder_1[i0_1, i1_1, i2_1, k, i4_1]
                    )

    mod = OperatorLegalizer(Matmul).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_softmax():
    @I.ir_module
    class Softmax:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 3, 16, 32), "float32") = R.softmax(x, axis=-2)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(softmax, (x,), (2, 3, 16, 32), dtype="float32")
            return gv

        @T.prim_func
        def softmax(
            rxplaceholder: T.Buffer[(2, 3, 16, 32), "float32"],
            T_softmax_norm: T.Buffer[(2, 3, 16, 32), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "softmax", "tir.noalias": True})
            T_softmax_maxelem = T.alloc_buffer([2, 3, 32], dtype="float32")
            T_softmax_exp = T.alloc_buffer([2, 3, 16, 32], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([2, 3, 32], dtype="float32")
            for i0, i1, i2, i3 in T.grid(2, 3, 32, 16):
                with T.block("T_softmax_maxelem"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, k, i2_1])
                    T.writes(T_softmax_maxelem[i0_1, i1_1, i2_1])
                    with T.init():
                        T_softmax_maxelem[i0_1, i1_1, i2_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[i0_1, i1_1, i2_1] = T.max(
                        T_softmax_maxelem[i0_1, i1_1, i2_1], rxplaceholder[i0_1, i1_1, k, i2_1]
                    )
            for i0, i1, i2, i3 in T.grid(2, 3, 16, 32):
                with T.block("T_softmax_exp"):
                    i0_2, i1_2, i2_2, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(
                        rxplaceholder[i0_2, i1_2, i2_2, i3_1], T_softmax_maxelem[i0_2, i1_2, i3_1]
                    )
                    T.writes(T_softmax_exp[i0_2, i1_2, i2_2, i3_1])
                    T_softmax_exp[i0_2, i1_2, i2_2, i3_1] = T.exp(
                        rxplaceholder[i0_2, i1_2, i2_2, i3_1] - T_softmax_maxelem[i0_2, i1_2, i3_1],
                        dtype="float32",
                    )
            for i0_3, i1_3, i2_3, i3 in T.grid(2, 3, 32, 16):
                with T.block("T_softmax_expsum"):
                    i0_4, i1_4, i2_4, k = T.axis.remap("SSSR", [i0_3, i1_3, i2_3, i3])
                    T.reads(T_softmax_exp[i0_4, i1_4, k, i2_4])
                    T.writes(T_softmax_expsum[i0_4, i1_4, i2_4])
                    with T.init():
                        T_softmax_expsum[i0_4, i1_4, i2_4] = T.float32(0)
                    T_softmax_expsum[i0_4, i1_4, i2_4] = (
                        T_softmax_expsum[i0_4, i1_4, i2_4] + T_softmax_exp[i0_4, i1_4, k, i2_4]
                    )
            for i0_5, i1_5, i2_5, i3 in T.grid(2, 3, 16, 32):
                with T.block("T_softmax_norm"):
                    i0_6, i1_6, i2_6, i3_2 = T.axis.remap("SSSS", [i0_5, i1_5, i2_5, i3])
                    T.reads(
                        T_softmax_exp[i0_6, i1_6, i2_6, i3_2], T_softmax_expsum[i0_6, i1_6, i3_2]
                    )
                    T.writes(T_softmax_norm[i0_6, i1_6, i2_6, i3_2])
                    T.block_attr({"axis": 2})
                    T_softmax_norm[i0_6, i1_6, i2_6, i3_2] = (
                        T_softmax_exp[i0_6, i1_6, i2_6, i3_2] / T_softmax_expsum[i0_6, i1_6, i3_2]
                    )

    mod = OperatorLegalizer(Softmax).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sum():
    @I.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((1, 3), "float32") = R.sum(x, axis=[1, 3])
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sum, (x,), (1, 3), dtype="float32")
            return gv

        @T.prim_func
        def sum(
            rxplaceholder: T.Buffer[(1, 2, 3, 4), "float32"],
            rxplaceholder_red: T.Buffer[(1, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "sum", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(1, 3, 2, 4):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k1, k3 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, k1, ax1, k3])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = (
                        rxplaceholder_red[ax0, ax1] + rxplaceholder[ax0, k1, ax1, k3]
                    )

    mod = OperatorLegalizer(Sum).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_mean():
    @I.ir_module
    class Mean:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((1, 3), "float32") = R.mean(x, axis=[1, 3])
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sum, (x,), (1, 3), dtype="float32")
            gv1 = R.call_tir(divide, (gv,), (1, 3), dtype="float32")
            return gv1

        @T.prim_func
        def sum(
            rxplaceholder: T.Buffer[(1, 2, 3, 4), "float32"],
            rxplaceholder_red: T.Buffer[(1, 3), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "sum", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(1, 3, 2, 4):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k1, k3 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, k1, ax1, k3])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = (
                        rxplaceholder_red[ax0, ax1] + rxplaceholder[ax0, k1, ax1, k3]
                    )

        @T.prim_func
        def divide(
            rxplaceholder: T.Buffer[(1, 3), "float32"], T_divide: T.Buffer[(1, 3), "float32"]
        ) -> None:
            T.func_attr({"global_symbol": "divide", "tir.noalias": True})
            for i0, i1 in T.grid(1, 3):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder[ax0, ax1] * T.float32(0.125)

    mod = OperatorLegalizer(Mean).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_conv2d()
    test_add()
    test_subtract()
    test_multiply()
    test_divide()
    test_floor_divide()
    test_sin()
    test_cos()
    test_sqrt()
    test_relu()
    test_gelu()
    test_silu()
    test_reshape()
    test_reshape_dim_inference()
    test_transpose()
    test_concatenate()
    test_cumsum()
    test_cumsum_without_specified_axis()
    test_expand_dims()
    test_trilu()
    test_cast()
    test_layer_norm()
    test_matmul_1_4()
    test_matmul_4_1()
    test_matmul_1_1()
    test_matmul_4_5()
    test_softmax()
    test_sum()
    test_mean()
