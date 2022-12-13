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
from tvm.script import ir as I, relax as R, tir as T
import tvm.testing


def test_conv2d():
    @I.ir_module
    class Conv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, kernel_size=[3, 3])
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
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(4), T.int64(3), T.int64(3), T.int64(3)), "float32"],
            conv2d_nchw: T.Buffer[(T.int64(2), T.int64(4), T.int64(26), T.int64(26)), "float32"],
        ):
            T.func_attr({"global_symbol": "conv2d", "tir.noalias": True})
            pad_temp = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(28), T.int64(28)], dtype="float32"
            )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(
                T.int64(2), T.int64(4), T.int64(26), T.int64(26), T.int64(3), T.int64(3), T.int64(3)
            ):
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


def test_conv2d_with_out_dtype():
    @I.ir_module
    class Conv2d:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float16", ndim=4):
            gv: R.Tensor((2, 4, 26, 26), "float16") = R.nn.conv2d(
                x, w, kernel_size=[3, 3], out_dtype="float16"
            )
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float16", ndim=4):
            gv = R.call_tir(conv2d, (x, w), (2, 4, 26, 26), dtype="float16")
            return gv

        @T.prim_func
        def conv2d(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(4), T.int64(3), T.int64(3), T.int64(3)), "float32"],
            conv2d_nchw: T.Buffer[(T.int64(2), T.int64(4), T.int64(26), T.int64(26)), "float16"],
        ):
            T.func_attr({"global_symbol": "conv2d", "tir.noalias": True})
            pad_temp = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(28), T.int64(28)], dtype="float32"
            )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(
                T.int64(2), T.int64(4), T.int64(26), T.int64(26), T.int64(3), T.int64(3), T.int64(3)
            ):
                with T.block("conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = T.axis.remap(
                        "SSSSRRR", [i0, i1, i2, i3, i4, i5, i6]
                    )
                    T.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
                    T.writes(conv2d_nchw[nn, ff, yy, xx])
                    with T.init():
                        conv2d_nchw[nn, ff, yy, xx] = T.float16(0)
                    conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + T.Cast(
                        "float16", pad_temp[nn, rc, yy + ry, xx + rx]
                    ) * T.Cast("float16", rxplaceholder_1[ff, rc, ry, rx])

    mod = OperatorLegalizer(Conv2d).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_add():
    @I.ir_module
    class Add:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.add(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(add, (x, y), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def add(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_add: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "add", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
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
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.subtract(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(subtract, (x, y), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def subtract(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_subtract: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "subtract", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
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
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.multiply(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32"),
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(multiply, (x, y), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def multiply(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_multiply: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "multiply", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
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
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.divide(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(divide, (x, y), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def divide(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(1)), "float32"],
            T_divide: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "divide", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, T.int64(0)])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder[ax0, ax1] / rxplaceholder_1[ax0, T.int64(0)]

    mod = OperatorLegalizer(Divide).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_floor_divide():
    @I.ir_module
    class FloorDivide:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.floor_divide(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"), y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(floor_divide, (x, y), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def floor_divide(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(1)), "float32"],
            T_floor_divide: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "floor_divide", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_floor_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, T.int64(0)])
                    T.writes(T_floor_divide[ax0, ax1])
                    T_floor_divide[ax0, ax1] = T.floor(
                        rxplaceholder[ax0, ax1] / rxplaceholder_1[ax0, T.int64(0)], dtype="float32"
                    )

    mod = OperatorLegalizer(FloorDivide).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_log():
    @I.ir_module
    class Log:
        @R.function
        def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.log(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def expected(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "expected"})
            gv = R.call_tir(log, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def log(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "log", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.log(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Log).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_negative():
    @I.ir_module
    class Negative:
        @R.function
        def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.negative(x)
            return gv


    @I.ir_module
    class Expected:
        @R.function
        def expected(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "expected"})
            gv = R.call_tir(negative, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def negative(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "negative", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = rxplaceholder[i0_1, i1_1] * T.float32(-1)

    mod = OperatorLegalizer(Negative).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sin():
    @I.ir_module
    class Sin:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.sin(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sin, (x,), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def sin(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "sin", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
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
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.cos(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(cos, (x,), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def cos(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "cos", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.cos(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Cos).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tanh():
    @I.ir_module
    class Tanh:
        @R.function
        def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.tanh(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def expected(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "expected"})
            gv = R.call_tir(tanh, (x,), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def tanh(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "tanh", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.tanh(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Tanh).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sqrt():
    @I.ir_module
    class Sqrt:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.sqrt(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sqrt, (x,), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def sqrt(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "sqrt", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sqrt(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Sqrt).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sigmoid():
    @I.ir_module
    class Sigmoid:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.sigmoid(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sigmoid, (x,), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def sigmoid(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"]
        ):
            T.func_attr({"global_symbol": "sigmoid", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1], dtype="float32")

    mod = OperatorLegalizer(Sigmoid).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_less():
    @I.ir_module
    class Less:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "bool", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "bool") = R.less(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32"),
            y: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "bool", ndim=2):
            gv = R.call_tir(less, (x, y), (2, 3), dtype="bool")
            return gv

        @T.prim_func
        def less(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_less: T.Buffer[(T.int64(2), T.int64(3)), "bool"]
        ):
            T.func_attr({"global_symbol": "less", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_less"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_less[ax0, ax1])
                    T_less[ax0, ax1] = rxplaceholder[ax0, ax1] < rxplaceholder_1[ax0, ax1]

    mod = OperatorLegalizer(Less).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_relu():
    @I.ir_module
    class Relu:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.nn.relu(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(relu, (x,), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def relu(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "relu", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
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
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.nn.gelu(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(gelu, (x,), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def gelu(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_multiply: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "gelu", "tir.noalias": True})
            T_multiply_1 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            compute = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_multiply_2 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_add = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_multiply_1[ax0, ax1])
                    T_multiply_1[ax0, ax1] = rxplaceholder[ax0, ax1] * T.float32(
                        0.70710678118654757
                    )
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_1[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.erf(T_multiply_1[i0_1, i1_1], dtype="float32")
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_1"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(compute[ax0, ax1])
                    T.writes(T_multiply_2[ax0, ax1])
                    T_multiply_2[ax0, ax1] = compute[ax0, ax1] * T.float32(0.5)
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_2[ax0, ax1])
                    T.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = T.float32(0.5) + T_multiply_2[ax0, ax1]
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_2"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], T_add[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * T_add[ax0, ax1]

    mod = OperatorLegalizer(Gelu).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_silu():
    @I.ir_module
    class Silu:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.nn.silu(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((T.int64(2), T.int64(3)), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(sigmoid, (x,), (T.int64(2), T.int64(3)), dtype="float32")
            gv1 = R.call_tir(multiply, (x, gv), (T.int64(2), T.int64(3)), dtype="float32")
            return gv1

        @T.prim_func
        def sigmoid(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            compute: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "sigmoid", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1], dtype="float32")

        @T.prim_func
        def multiply(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_multiply: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ) -> None:
            T.func_attr({"global_symbol": "multiply", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
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
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"],
            T_reshape: T.Buffer[(T.int64(8), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "reshape", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(8), T.int64(3)):
                with T.block("T_reshape"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(
                        rxplaceholder[
                            T.int64(0),
                            (ax0 * T.int64(3) + ax1) % T.int64(24) // T.int64(12),
                            (ax0 * T.int64(3) + ax1) % T.int64(12) // T.int64(4),
                            (ax0 * T.int64(3) + ax1) % T.int64(4),
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        T.int64(0),
                        (ax0 * T.int64(3) + ax1) % T.int64(24) // T.int64(12),
                        (ax0 * T.int64(3) + ax1) % T.int64(12) // T.int64(4),
                        (ax0 * T.int64(3) + ax1) % T.int64(4),
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
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"],
            T_reshape: T.Buffer[(T.int64(8), T.int64(1), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "reshape", "tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(8), T.int64(1), T.int64(3)):
                with T.block("T_reshape"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(
                        rxplaceholder[
                            T.int64(0),
                            (ax0 * T.int64(3) + ax1 * T.int64(3) + ax2)
                            % T.int64(24)
                            // T.int64(12),
                            (ax0 * T.int64(3) + ax1 * T.int64(3) + ax2) % T.int64(12) // T.int64(4),
                            (ax0 * T.int64(3) + ax1 * T.int64(3) + ax2) % T.int64(4),
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1, ax2])
                    T_reshape[ax0, ax1, ax2] = rxplaceholder[
                        T.int64(0),
                        (ax0 * T.int64(3) + ax1 * T.int64(3) + ax2) % T.int64(24) // T.int64(12),
                        (ax0 * T.int64(3) + ax1 * T.int64(3) + ax2) % T.int64(12) // T.int64(4),
                        (ax0 * T.int64(3) + ax1 * T.int64(3) + ax2) % T.int64(4),
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
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"],
            T_transpose: T.Buffer[(T.int64(2), T.int64(4), T.int64(3), T.int64(1)), "float32"],
        ):
            T.func_attr({"global_symbol": "transpose", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(4), T.int64(3), T.int64(1)):
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
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(1), T.int64(3), T.int64(3)), "float32"],
            rxplaceholder_2: T.Buffer[(T.int64(1), T.int64(4), T.int64(3)), "float32"],
            T_concat: T.Buffer[(T.int64(1), T.int64(9), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "concatenate", "tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(9), T.int64(3)):
                with T.block("T_concat"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(
                        rxplaceholder_2[ax0, ax1 - T.int64(5), ax2],
                        rxplaceholder_1[ax0, ax1 - T.int64(2), ax2],
                        rxplaceholder[ax0, ax1, ax2],
                    )
                    T.writes(T_concat[ax0, ax1, ax2])
                    T_concat[ax0, ax1, ax2] = T.if_then_else(
                        T.int64(5) <= ax1,
                        rxplaceholder_2[ax0, ax1 - T.int64(5), ax2],
                        T.if_then_else(
                            T.int64(2) <= ax1,
                            rxplaceholder_1[ax0, ax1 - T.int64(2), ax2],
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
            var_rxplaceholder: T.handle,
            out_buf: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "float32"],
        ):
            T.func_attr({"global_symbol": "cumsum", "tir.noalias": True})
            rxplaceholder = T.match_buffer(
                var_rxplaceholder,
                [T.int64(2), T.int64(3), T.int64(4)],
                dtype="float32",
                offset_factor=1,
            )
            with T.block("cumsum_generic"):
                T.reads(
                    rxplaceholder[
                        T.int64(0) : T.int64(2), T.int64(0) : T.int64(3), T.int64(0) : T.int64(4)
                    ]
                )
                T.writes(
                    out_buf[
                        T.int64(0) : T.int64(2), T.int64(0) : T.int64(3), T.int64(0) : T.int64(4)
                    ]
                )
                for fused in T.parallel(T.int64(8)):
                    out_buf[
                        (fused // T.int64(4) * T.int64(3) * T.int64(4) + fused % T.int64(4))
                        // T.int64(4)
                        // T.int64(3),
                        (fused // T.int64(4) * T.int64(3) * T.int64(4) + fused % T.int64(4))
                        // T.int64(4)
                        % T.int64(3),
                        (fused // T.int64(4) * T.int64(3) * T.int64(4) + fused % T.int64(4))
                        % T.int64(4),
                    ] = rxplaceholder[
                        (fused // T.int64(4) * T.int64(3) * T.int64(4) + fused % T.int64(4))
                        // T.int64(4)
                        // T.int64(3),
                        (fused // T.int64(4) * T.int64(3) * T.int64(4) + fused % T.int64(4))
                        // T.int64(4)
                        % T.int64(3),
                        (fused // T.int64(4) * T.int64(3) * T.int64(4) + fused % T.int64(4))
                        % T.int64(4),
                    ]
                    for v_k in T.serial(T.int64(2)):
                        out_buf[
                            (
                                fused // T.int64(4) * T.int64(3) * T.int64(4)
                                + fused % T.int64(4)
                                + (v_k + T.int64(1)) * T.int64(4)
                            )
                            // T.int64(4)
                            // T.int64(3),
                            (
                                fused // T.int64(4) * T.int64(3) * T.int64(4)
                                + fused % T.int64(4)
                                + (v_k + T.int64(1)) * T.int64(4)
                            )
                            // T.int64(4)
                            % T.int64(3),
                            (
                                fused // T.int64(4) * T.int64(3) * T.int64(4)
                                + fused % T.int64(4)
                                + (v_k + T.int64(1)) * T.int64(4)
                            )
                            % T.int64(4),
                        ] = (
                            out_buf[
                                (
                                    fused // T.int64(4) * T.int64(3) * T.int64(4)
                                    + fused % T.int64(4)
                                    + (v_k + T.int64(1) - T.int64(1)) * T.int64(4)
                                )
                                // T.int64(4)
                                // T.int64(3),
                                (
                                    fused // T.int64(4) * T.int64(3) * T.int64(4)
                                    + fused % T.int64(4)
                                    + (v_k + T.int64(1) - T.int64(1)) * T.int64(4)
                                )
                                // T.int64(4)
                                % T.int64(3),
                                (
                                    fused // T.int64(4) * T.int64(3) * T.int64(4)
                                    + fused % T.int64(4)
                                    + (v_k + T.int64(1) - T.int64(1)) * T.int64(4)
                                )
                                % T.int64(4),
                            ]
                            + rxplaceholder[
                                (
                                    fused // T.int64(4) * T.int64(3) * T.int64(4)
                                    + fused % T.int64(4)
                                    + (v_k + T.int64(1)) * T.int64(4)
                                )
                                // T.int64(4)
                                // T.int64(3),
                                (
                                    fused // T.int64(4) * T.int64(3) * T.int64(4)
                                    + fused % T.int64(4)
                                    + (v_k + T.int64(1)) * T.int64(4)
                                )
                                // T.int64(4)
                                % T.int64(3),
                                (
                                    fused // T.int64(4) * T.int64(3) * T.int64(4)
                                    + fused % T.int64(4)
                                    + (v_k + T.int64(1)) * T.int64(4)
                                )
                                % T.int64(4),
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
        def cumsum(var_rxplaceholder: T.handle, out_buf: T.Buffer[T.int64(24), "float32"]):
            T.func_attr({"global_symbol": "cumsum", "tir.noalias": True})
            rxplaceholder = T.match_buffer(
                var_rxplaceholder,
                [T.int64(2), T.int64(3), T.int64(4)],
                dtype="float32",
                offset_factor=1,
            )
            with T.block("cumsum_generic"):
                T.reads(
                    rxplaceholder[
                        T.int64(0) : T.int64(2), T.int64(0) : T.int64(3), T.int64(0) : T.int64(4)
                    ]
                )
                T.writes(out_buf[T.int64(0) : T.int64(24)])
                for fused in T.parallel(T.int64(1)):
                    out_buf[fused * T.int64(24)] = rxplaceholder[
                        fused * T.int64(24) // T.int64(4) // T.int64(3),
                        fused * T.int64(24) // T.int64(4) % T.int64(3),
                        fused * T.int64(24) % T.int64(4),
                    ]
                    for v_k in T.serial(T.int64(23)):
                        out_buf[fused * T.int64(24) + (v_k + T.int64(1))] = (
                            out_buf[fused * T.int64(24) + (v_k + T.int64(1) - T.int64(1))]
                            + rxplaceholder[
                                (fused * T.int64(24) + (v_k + T.int64(1)))
                                // T.int64(4)
                                // T.int64(3),
                                (fused * T.int64(24) + (v_k + T.int64(1)))
                                // T.int64(4)
                                % T.int64(3),
                                (fused * T.int64(24) + (v_k + T.int64(1))) % T.int64(4),
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
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "float32"],
            compute: T.Buffer[
                (
                    T.int64(2),
                    T.int64(1),
                    T.int64(1),
                    T.int64(1),
                    T.int64(3),
                    T.int64(1),
                    T.int64(4),
                    T.int64(1),
                ),
                "float32",
            ],
        ):
            T.func_attr({"global_symbol": "expand_dims", "tir.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(
                T.int64(2),
                T.int64(1),
                T.int64(1),
                T.int64(1),
                T.int64(3),
                T.int64(1),
                T.int64(4),
                T.int64(1),
            ):
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
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "float32"],
            trilu: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "float32"],
        ):
            T.func_attr({"global_symbol": "trilu", "tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
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
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "float32"],
            compute: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "int32"],
        ):
            T.func_attr({"global_symbol": "cast", "tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with T.block("compute"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(compute[i0_1, i1_1, i2_1])
                    compute[i0_1, i1_1, i2_1] = T.Cast("int32", rxplaceholder[i0_1, i1_1, i2_1])

    mod = OperatorLegalizer(Cast).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_take():
    @I.ir_module
    class Take:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), "float32"), indices: R.Tensor((3, 4, 2), "int32")
        ) -> R.Tensor(None, "float32", ndim=5):
            gv: R.Tensor((2, 3, 4, 2, 4), "float32") = R.take(x, indices, axis=1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), "float32"), indices: R.Tensor((3, 4, 2), "int32")
        ) -> R.Tensor(None, "float32", ndim=5):
            gv = R.call_tir(take, (x, indices), (2, 3, 4, 2, 4), dtype="float32")
            return gv

        @T.prim_func
        def take(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(3), T.int64(4), T.int64(2)), "int32"],
            T_take: T.Buffer[
                (T.int64(2), T.int64(3), T.int64(4), T.int64(2), T.int64(4)), "float32"
            ],
        ):
            T.func_attr({"global_symbol": "take", "tir.noalias": True})
            for i0, i1, i2, i3, i4 in T.grid(
                T.int64(2), T.int64(3), T.int64(4), T.int64(2), T.int64(4)
            ):
                with T.block("T_take"):
                    ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(
                        rxplaceholder[
                            ax0,
                            T.min(
                                T.max(T.int64(0), T.Cast("int64", rxplaceholder_1[ax1, ax2, ax3])),
                                T.int64(2),
                            ),
                            ax4,
                        ],
                        rxplaceholder_1[ax1, ax2, ax3],
                    )
                    T.writes(T_take[ax0, ax1, ax2, ax3, ax4])
                    T_take[ax0, ax1, ax2, ax3, ax4] = rxplaceholder[
                        ax0,
                        T.min(
                            T.max(T.int64(0), T.Cast("int64", rxplaceholder_1[ax1, ax2, ax3])),
                            T.int64(2),
                        ),
                        ax4,
                    ]

    mod = OperatorLegalizer(Take).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full():
    @I.ir_module
    class Full:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((T.int64(2), T.int64(3)), "float32") = R.full(
                v, (T.int64(2), T.int64(3)), dtype="float32"
            )
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(full, (v,), (T.int64(2), T.int64(3)), dtype="float32")
            return gv

        @T.prim_func
        def full(
            rxplaceholder: T.Buffer[(), "int32"],
            T_full: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "full", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.Cast("float32", rxplaceholder[()])

    mod = OperatorLegalizer(Full).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_like():
    @I.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.full_like(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "main"})
            gv = R.call_tir(full_like, (x, y), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def full_like(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], rxplaceholder_1: T.Buffer[(), "float32"], T_full_like: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "full_like", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full_like"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_1[()])
                    T.writes(T_full_like[ax0, ax1])
                    T_full_like[ax0, ax1] = rxplaceholder_1[()]

    mod = OperatorLegalizer(FullLike).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_ones_like():
    @I.ir_module
    class OnesLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.ones_like(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "main"})
            gv = R.call_tir(full_like, (x, R.const(1.0)), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def full_like(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], rxplaceholder_1: T.Buffer[(), "float32"], T_full_like: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "full_like", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full_like"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_1[()])
                    T.writes(T_full_like[ax0, ax1])
                    T_full_like[ax0, ax1] = rxplaceholder_1[()]

    mod = OperatorLegalizer(OnesLike).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_zeros_like():
    @I.ir_module
    class ZerosLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.zeros_like(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "main"})
            gv = R.call_tir(full_like, (x, R.const(0.0)), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def full_like(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], rxplaceholder_1: T.Buffer[(), "float32"], T_full_like: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "full_like", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full_like"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_1[()])
                    T.writes(T_full_like[ax0, ax1])
                    T_full_like[ax0, ax1] = rxplaceholder_1[()]

    mod = OperatorLegalizer(ZerosLike).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_ones():
    @I.ir_module
    class Ones:
        @R.function
        def main() -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.ones((2, 3))
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "main"})
            gv = R.call_tir(full, (R.const(1.0),), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer[(), "float32"], T_full: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "full", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]

    mod = OperatorLegalizer(Ones).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_zeros():
    @I.ir_module
    class Zeros:
        @R.function
        def main() -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.zeros((2, 3))
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "expected"})
            gv = R.call_tir(full, (R.const(0.0),), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer[(), "float32"], T_full: T.Buffer[(T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "full", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]

    mod = OperatorLegalizer(Zeros).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_collapse_sum_like():
    @I.ir_module
    class CollapseSumLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((1, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((1, 3), "float32") = R.collapse_sum_like(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((1, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "expected"})
            gv = R.call_tir(collapse_sum, (x,), (1, 3), dtype="float32")
            return gv

        @T.prim_func
        def collapse_sum(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], rxplaceholder_red: T.Buffer[(T.int64(1), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "collapse_sum", "tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(3), T.int64(2)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k0 = T.axis.remap("SSR", [i0, i1, i2])
                    T.reads(rxplaceholder[k0, ax1])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax1]

    mod = OperatorLegalizer(CollapseSumLike).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_collapse_sum_to():
    @I.ir_module
    class CollapseSumTo:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((1, 3), "float32") = R.collapse_sum_to(x, (1, 3))
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=2):
            R.func_attr({"global_symbol": "main"})
            gv = R.call_tir(collapse_sum, (x,), (1, 3), dtype="float32")
            return gv

        @T.prim_func
        def collapse_sum(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"], rxplaceholder_red: T.Buffer[(T.int64(1), T.int64(3)), "float32"]):
            T.func_attr({"global_symbol": "collapse_sum", "tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(3), T.int64(2)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k0 = T.axis.remap("SSR", [i0, i1, i2])
                    T.reads(rxplaceholder[k0, ax1])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax1]

    mod = OperatorLegalizer(CollapseSumTo).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_broadcast_to():
    @I.ir_module
    class BroadcastTo:
        @R.function
        def main(x: R.Tensor((2, 1, 3), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((4, 2, 5, 3), "float32") = R.broadcast_to(x, (4, 2, 5, 3))
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(broadcast_to, (x,), (4, 2, 5, 3), dtype="float32")
            return gv

        @T.prim_func
        def broadcast_to(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(1), T.int64(3)), "float32"],
            T_broadcast_to: T.Buffer[(T.int64(4), T.int64(2), T.int64(5), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "broadcast_to", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(2), T.int64(5), T.int64(3)):
                with T.block("T_broadcast_to"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax1, T.int64(0), ax3])
                    T.writes(T_broadcast_to[ax0, ax1, ax2, ax3])
                    T_broadcast_to[ax0, ax1, ax2, ax3] = rxplaceholder[ax1, T.int64(0), ax3]

    mod = OperatorLegalizer(BroadcastTo).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_where():
    @I.ir_module
    class Where:
        @R.function
        def main(
            condition: R.Tensor((2, 1), "bool"),
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((2, 3), "float32") = R.where(condition, x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            condition: R.Tensor((2, 1), "bool"),
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((2, 1), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_tir(where, (condition, x, y), (2, 3), dtype="float32")
            return gv

        @T.prim_func
        def where(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(1)), "bool"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_2: T.Buffer[(T.int64(2), T.int64(1)), "float32"],
            T_where: T.Buffer[(T.int64(2), T.int64(3)), "float32"]
        ):
            T.func_attr({"global_symbol": "where", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_where"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, T.int64(0)], rxplaceholder_1[ax0, ax1], rxplaceholder_2[ax0, T.int64(0)])
                    T.writes(T_where[ax0, ax1])
                    T_where[ax0, ax1] = T.Select(
                        0 < T.Cast("int32", rxplaceholder[ax0, T.int64(0)]),
                        rxplaceholder_1[ax0, ax1],
                        rxplaceholder_2[ax0, T.int64(0)]
                    )

    mod = OperatorLegalizer(Where).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_strided_slice():
    @I.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((4, 9, 10, 3), "float32") = R.strided_slice(
                x,
                begin=[1, 0, 8],
                end=[8, 9, 0],
                strides=[2, 1, -3],
                axes=[0, 1, 3],
                slice_mode="end",
            )
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(strided_slice, (x,), (4, 9, 10, 3), dtype="float32")
            return gv

        @T.prim_func
        def strided_slice(
            rxplaceholder: T.Buffer[(T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"],
            T_strided_slice_with_axes: T.Buffer[
                (T.int64(4), T.int64(9), T.int64(10), T.int64(3)), "float32"
            ],
        ):
            T.func_attr({"global_symbol": "strided_slice", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(9), T.int64(10), T.int64(3)):
                with T.block("T_strided_slice_with_axes"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(
                        rxplaceholder[
                            ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)
                        ]
                    )
                    T.writes(T_strided_slice_with_axes[ax0, ax1, ax2, ax3])
                    T_strided_slice_with_axes[ax0, ax1, ax2, ax3] = rxplaceholder[
                        ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)
                    ]

    mod = OperatorLegalizer(StridedSlice).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d():
    @I.ir_module
    class MaxPool2D:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((4, 6, 56, 56), "float32") = R.nn.max_pool2d(
                x,
                pool_size=[3, 3],
                strides=[2, 2],
                dilation=[1, 1],
                padding=[1, 1, 1, 1],
                layout="NCHW",
            )
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(pool2d, (x,), (4, 6, 56, 56), dtype="float32")
            return gv

        @T.prim_func
        def pool2d(
            rxplaceholder: T.Buffer[
                (T.int64(4), T.int64(6), T.int64(112), T.int64(112)), "float32"
            ],
            pool_max: T.Buffer[(T.int64(4), T.int64(6), T.int64(56), T.int64(56)), "float32"],
        ):
            T.func_attr({"global_symbol": "pool2d", "tir.noalias": True})
            pad_temp = T.alloc_buffer(
                [T.int64(4), T.int64(6), T.int64(114), T.int64(114)], dtype="float32"
            )
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(6), T.int64(114), T.int64(114)):
                with T.block("pad_temp"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2 - T.int64(1), ax3 - T.int64(1)])
                    T.writes(pad_temp[ax0, ax1, ax2, ax3])
                    pad_temp[ax0, ax1, ax2, ax3] = T.if_then_else(
                        T.int64(1) <= ax2
                        and ax2 < T.int64(113)
                        and T.int64(1) <= ax3
                        and ax3 < T.int64(113),
                        rxplaceholder[ax0, ax1, ax2 - T.int64(1), ax3 - T.int64(1)],
                        T.float32(-3.4028234663852886e38),
                        dtype="float32",
                    )
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(4), T.int64(6), T.int64(56), T.int64(56), T.int64(3), T.int64(3)
            ):
                with T.block("pool_max"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(pad_temp[ax0, ax1, ax2 * T.int64(2) + rv0, ax3 * T.int64(2) + rv1])
                    T.writes(pool_max[ax0, ax1, ax2, ax3])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                    with T.init():
                        pool_max[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e38)
                    pool_max[ax0, ax1, ax2, ax3] = T.max(
                        pool_max[ax0, ax1, ax2, ax3],
                        pad_temp[ax0, ax1, ax2 * T.int64(2) + rv0, ax3 * T.int64(2) + rv1],
                    )

    mod = OperatorLegalizer(MaxPool2D).transform()
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
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.layer_norm(x, gamma, beta, axis=[-2, -1])
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
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(4), T.int64(5)), "float32"],
            rxplaceholder_2: T.Buffer[(T.int64(4), T.int64(5)), "float32"],
            T_add: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"],
        ):
            # function attr dict
            T.func_attr({"global_symbol": "layer_norm", "tir.noalias": True})
            # body
            # with T.block("root")
            rxplaceholder_red = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(1), T.int64(1)], dtype="float32"
            )
            T_divide = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(1), T.int64(1)], dtype="float32"
            )
            T_subtract = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32"
            )
            T_subtract_1 = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32"
            )
            T_subtract_2 = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32"
            )
            T_multiply = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32"
            )
            T_multiply_red = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(1), T.int64(1)], dtype="float32"
            )
            T_divide_1 = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(1), T.int64(1)], dtype="float32"
            )
            T_add_1 = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(1), T.int64(1)], dtype="float32"
            )
            compute = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(1), T.int64(1)], dtype="float32"
            )
            T_divide_2 = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32"
            )
            T_multiply_1 = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32"
            )
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(2), T.int64(3), T.int64(1), T.int64(1), T.int64(4), T.int64(5)
            ):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k2, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[ax0, ax1, k2, k3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = (
                        rxplaceholder_red[ax0, ax1, ax2, ax3] + rxplaceholder[ax0, ax1, k2, k3]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = rxplaceholder_red[
                        ax0, ax1, ax2, ax3
                    ] * T.float32(0.050000000000000003)
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(
                        rxplaceholder[ax0, ax1, ax2, ax3],
                        T_divide[ax0, ax1, T.int64(0), T.int64(0)],
                    )
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = (
                        rxplaceholder[ax0, ax1, ax2, ax3]
                        - T_divide[ax0, ax1, T.int64(0), T.int64(0)]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_subtract_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(
                        rxplaceholder[ax0, ax1, ax2, ax3],
                        T_divide[ax0, ax1, T.int64(0), T.int64(0)],
                    )
                    T.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                    T_subtract_1[ax0, ax1, ax2, ax3] = (
                        rxplaceholder[ax0, ax1, ax2, ax3]
                        - T_divide[ax0, ax1, T.int64(0), T.int64(0)]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_subtract_2"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(
                        rxplaceholder[ax0, ax1, ax2, ax3],
                        T_divide[ax0, ax1, T.int64(0), T.int64(0)],
                    )
                    T.writes(T_subtract_2[ax0, ax1, ax2, ax3])
                    T_subtract_2[ax0, ax1, ax2, ax3] = (
                        rxplaceholder[ax0, ax1, ax2, ax3]
                        - T_divide[ax0, ax1, T.int64(0), T.int64(0)]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_subtract_1[ax0, ax1, ax2, ax3], T_subtract_2[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = (
                        T_subtract_1[ax0, ax1, ax2, ax3] * T_subtract_2[ax0, ax1, ax2, ax3]
                    )
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(2), T.int64(3), T.int64(1), T.int64(1), T.int64(4), T.int64(5)
            ):
                with T.block("T_multiply_red"):
                    ax0, ax1, ax2, ax3, k2, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(T_multiply[ax0, ax1, k2, k3])
                    T.writes(T_multiply_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        T_multiply_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    T_multiply_red[ax0, ax1, ax2, ax3] = (
                        T_multiply_red[ax0, ax1, ax2, ax3] + T_multiply[ax0, ax1, k2, k3]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_divide_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_multiply_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide_1[ax0, ax1, ax2, ax3])
                    T_divide_1[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] * T.float32(
                        0.050000000000000003
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_divide_1[ax0, ax1, ax2, ax3])
                    T.writes(T_add_1[ax0, ax1, ax2, ax3])
                    T_add_1[ax0, ax1, ax2, ax3] = T_divide_1[ax0, ax1, ax2, ax3] + T.float32(
                        1.0000000000000001e-05
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("compute"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_add_1[i0_1, i1_1, i2_1, i3_1])
                    T.writes(compute[i0_1, i1_1, i2_1, i3_1])
                    compute[i0_1, i1_1, i2_1, i3_1] = T.sqrt(
                        T_add_1[i0_1, i1_1, i2_1, i3_1], dtype="float32"
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_divide_2"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(
                        T_subtract[ax0, ax1, ax2, ax3], compute[ax0, ax1, T.int64(0), T.int64(0)]
                    )
                    T.writes(T_divide_2[ax0, ax1, ax2, ax3])
                    T_divide_2[ax0, ax1, ax2, ax3] = (
                        T_subtract[ax0, ax1, ax2, ax3] / compute[ax0, ax1, T.int64(0), T.int64(0)]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax2, ax3], T_divide_2[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply_1[ax0, ax1, ax2, ax3])
                    T_multiply_1[ax0, ax1, ax2, ax3] = (
                        rxplaceholder_1[ax2, ax3] * T_divide_2[ax0, ax1, ax2, ax3]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
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
            gv: R.Tensor((2, 3, 5), "float32") = R.nn.matmul(x, y)
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
            rxplaceholder: T.Buffer[T.int64(4), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"],
            matmul: T.Buffer[(T.int64(2), T.int64(3), T.int64(5)), "float32"],
        ):
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(5), T.int64(4)):
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
            gv: R.Tensor((2, 3, 4), "float32") = R.nn.matmul(x, y)
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
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"],
            rxplaceholder_1: T.Buffer[T.int64(5), "float32"],
            matmul: T.Buffer[(T.int64(2), T.int64(3), T.int64(4)), "float32"],
        ):
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
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
            gv: R.Tensor((), "float32") = R.nn.matmul(x, y)
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
            rxplaceholder: T.Buffer[T.int64(4), "float32"],
            rxplaceholder_1: T.Buffer[T.int64(4), "float32"],
            matmul: T.Buffer[(), "float32"],
        ):
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0 in T.serial(T.int64(4)):
                with T.block("matmul"):
                    k = T.axis.reduce(T.int64(4), i0)
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
            gv: R.Tensor((6, 2, 3, 4, 7), "float32") = R.nn.matmul(x, y)
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
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"],
            rxplaceholder_1: T.Buffer[
                (T.int64(6), T.int64(2), T.int64(3), T.int64(5), T.int64(7)), "float32"
            ],
            matmul: T.Buffer[
                (T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7)), "float32"
            ],
        ):
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7), T.int64(5)
            ):
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


def test_matmul_4_5_with_out_dtype():
    @I.ir_module
    class Matmul:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((6, 2, 3, 5, 7), "float32")
        ) -> R.Tensor(None, "float16", ndim=5):
            gv: R.Tensor((6, 2, 3, 4, 7), "float16") = R.nn.matmul(x, y, out_dtype="float16")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((6, 2, 3, 5, 7), "float32")
        ) -> R.Tensor(None, "float16", ndim=5):
            gv = R.call_tir(matmul, (x, y), (6, 2, 3, 4, 7), dtype="float16")
            return gv

        @T.prim_func
        def matmul(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"],
            rxplaceholder_1: T.Buffer[
                (T.int64(6), T.int64(2), T.int64(3), T.int64(5), T.int64(7)), "float32"
            ],
            matmul: T.Buffer[
                (T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7)), "float16"
            ],
        ):
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7), T.int64(5)
            ):
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
                        matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = T.float16(0)
                    matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = matmul[
                        i0_1, i1_1, i2_1, i3_1, i4_1
                    ] + T.Cast("float16", rxplaceholder[i1_1, i2_1, i3_1, k]) * T.Cast(
                        "float16", rxplaceholder_1[i0_1, i1_1, i2_1, k, i4_1]
                    )

    mod = OperatorLegalizer(Matmul).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_3_3_with_symbolic_broadcast_dim():
    a = tvm.tir.Var("a", dtype="int64")

    @I.ir_module
    class Matmul:
        @R.function
        def main(
            x: R.Tensor((a, 3, 4), "float32"), y: R.Tensor((1, 4, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            gv: R.Tensor((a, 3, 5), "float32") = R.nn.matmul(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((a, 3, 4), "float32"), y: R.Tensor((1, 4, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            gv = R.call_tir(matmul, (x, y), (a, 3, 5), dtype="float32")
            return gv

        @T.prim_func
        def matmul(
            var_rxplaceholder: T.handle,
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(4), T.int64(5)), "float32"],
            var_matmul: T.handle,
        ):
            T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
            a = T.var("int64")
            rxplaceholder_1 = T.match_buffer(
                var_rxplaceholder, [a, T.int64(3), T.int64(4)], dtype="float32"
            )
            matmul = T.match_buffer(var_matmul, [a, T.int64(3), T.int64(5)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, T.int64(3), T.int64(5), T.int64(4)):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[i0_1, i1_1, k], rxplaceholder[T.int64(0), k, i2_1])
                    T.writes(matmul[i0_1, i1_1, i2_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1] = (
                        matmul[i0_1, i1_1, i2_1]
                        + rxplaceholder_1[i0_1, i1_1, k] * rxplaceholder[T.int64(0), k, i2_1]
                    )

    mod = OperatorLegalizer(Matmul).transform()
    # TVMScript and Relax function now have limited support on understanding symbolic variables. So
    # at this moment we only compare the the generated PrimFunc.
    tvm.ir.assert_structural_equal(mod["matmul"], Expected["matmul"])


def test_softmax():
    @I.ir_module
    class Softmax:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 3, 16, 32), "float32") = R.nn.softmax(x, axis=-2)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(softmax, (x,), (2, 3, 16, 32), dtype="float32")
            return gv

        @T.prim_func
        def softmax(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"],
            T_softmax_norm: T.Buffer[(T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"],
        ):
            T.func_attr({"global_symbol": "softmax", "tir.noalias": True})
            T_softmax_maxelem = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(32)], dtype="float32"
            )
            T_softmax_exp = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(16), T.int64(32)], dtype="float32"
            )
            T_softmax_expsum = T.alloc_buffer(
                [T.int64(2), T.int64(3), T.int64(32)], dtype="float32"
            )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with T.block("T_softmax_maxelem"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, k, i2_1])
                    T.writes(T_softmax_maxelem[i0_1, i1_1, i2_1])
                    with T.init():
                        T_softmax_maxelem[i0_1, i1_1, i2_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[i0_1, i1_1, i2_1] = T.max(
                        T_softmax_maxelem[i0_1, i1_1, i2_1], rxplaceholder[i0_1, i1_1, k, i2_1]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
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
            for i0_3, i1_3, i2_3, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with T.block("T_softmax_expsum"):
                    i0_4, i1_4, i2_4, k = T.axis.remap("SSSR", [i0_3, i1_3, i2_3, i3])
                    T.reads(T_softmax_exp[i0_4, i1_4, k, i2_4])
                    T.writes(T_softmax_expsum[i0_4, i1_4, i2_4])
                    with T.init():
                        T_softmax_expsum[i0_4, i1_4, i2_4] = T.float32(0)
                    T_softmax_expsum[i0_4, i1_4, i2_4] = (
                        T_softmax_expsum[i0_4, i1_4, i2_4] + T_softmax_exp[i0_4, i1_4, k, i2_4]
                    )
            for i0_5, i1_5, i2_5, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
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


def test_adaptive_avg_pool2d():
    @I.ir_module
    class AdaptiveAvgPool2D:
        @R.function
        def main(x: R.Tensor((2, 64, 7, 7), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 64, 1, 1), "float32") = R.nn.adaptive_avg_pool2d(x, output_size=[1, 1])
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 64, 7, 7), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(adaptive_pool, (x,), (2, 64, 1, 1), dtype="float32")
            return gv

        @T.prim_func
        def adaptive_pool(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(64), T.int64(7), T.int64(7)), "float32"],
            adaptive_pool_avg: T.Buffer[
                (T.int64(2), T.int64(64), T.int64(1), T.int64(1)), "float32"
            ],
        ):
            T.func_attr({"global_symbol": "adaptive_pool", "tir.noalias": True})
            adaptive_pool_sum = T.alloc_buffer(
                [T.int64(2), T.int64(64), T.int64(1), T.int64(1)], dtype="float32"
            )
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(2), T.int64(64), T.int64(1), T.int64(1), T.int64(7), T.int64(7)
            ):
                with T.block("adaptive_pool_sum"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[ax0, ax1, ax2 * T.int64(7) + rv0, ax3 * T.int64(7) + rv1])
                    T.writes(adaptive_pool_sum[ax0, ax1, ax2, ax3])
                    with T.init():
                        adaptive_pool_sum[ax0, ax1, ax2, ax3] = T.float32(0)
                    adaptive_pool_sum[ax0, ax1, ax2, ax3] = (
                        adaptive_pool_sum[ax0, ax1, ax2, ax3]
                        + rxplaceholder[ax0, ax1, ax2 * T.int64(7) + rv0, ax3 * T.int64(7) + rv1]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(64), T.int64(1), T.int64(1)):
                with T.block("adaptive_pool_avg"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(adaptive_pool_sum[ax0, ax1, ax2, ax3])
                    T.writes(adaptive_pool_avg[ax0, ax1, ax2, ax3])
                    T.block_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[ax0, ax1, ax2, ax3] = adaptive_pool_sum[
                        ax0, ax1, ax2, ax3
                    ] * T.float32(0.020408163265306121)

    mod = OperatorLegalizer(AdaptiveAvgPool2D).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cross_entropy():
    @I.ir_module
    class CrossEntropy:
        @R.function
        def main(predictions: R.Tensor((2, 3), "float32"), targets: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=0):
            gv: R.Tensor((), "float32") = R.nn.cross_entropy(predictions, targets)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor((2, 3), "float32"), targets: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=0):
            gv: R.Tensor((), "float32") = R.call_tir(cross_entropy, (predictions, targets), (), dtype="float32")
            return gv

        @T.prim_func
        def cross_entropy(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_multiply: T.Buffer[(), "float32"]
        ):
            T.func_attr({"global_symbol": "cross_entropy", "tir.noalias": True})
            compute = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_multiply_1 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_multiply_red = T.alloc_buffer([], dtype="float32")

            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.log(rxplaceholder[i0_1, i1_1], dtype="float32")

            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(compute[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_multiply_1[ax0, ax1])
                    T_multiply_1[ax0, ax1] = compute[ax0, ax1] * rxplaceholder_1[ax0, ax1]

            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_red"):
                    k0, k1 = T.axis.remap("RR", [i0, i1])
                    T.reads(T_multiply_1[k0, k1])
                    T.writes(T_multiply_red[()])
                    with T.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply_1[k0, k1]

            with T.block("T_multiply_1"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_red[()])
                T.writes(T_multiply[()])
                T_multiply[()] = T_multiply_red[()] * T.float32(-1)

    mod = OperatorLegalizer(CrossEntropy).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_softmax_cross_entropy():
    @I.ir_module
    class SoftmaxCrossEntropy:
        @R.function
        def main(predictions: R.Tensor((2, 3), "float32"), targets: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=0):
            gv: R.Tensor((), "float32") = R.nn.softmax_cross_entropy(predictions, targets)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor((2, 3), "float32"), targets: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=0):
            gv: R.Tensor((), "float32") = R.call_tir(softmax_cross_entropy, (predictions, targets), (), dtype="float32")
            return gv

        @T.prim_func
        def softmax_cross_entropy(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"],
            T_multiply: T.Buffer[(), "float32"]
        ):
            T.func_attr({"global_symbol": "softmax_cross_entropy", "tir.noalias": True})
            T_softmax_maxelem = T.alloc_buffer([T.int64(2)], dtype="float32")
            T_softmax_exp = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([T.int64(2)], dtype="float32")
            T_softmax_norm = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            compute = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_multiply_1 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_multiply_red = T.alloc_buffer([], dtype="float32")

            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_softmax_maxelem"):
                    i0_1, k = T.axis.remap("SR", [i0, i1])
                    T.reads(rxplaceholder[i0_1, k])
                    T.writes(T_softmax_maxelem[i0_1])
                    with T.init():
                        T_softmax_maxelem[i0_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_1] = T.max(T_softmax_maxelem[i0_1], rxplaceholder[i0_1, k])

            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_softmax_exp"):
                    i0_2, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_2, i1_1], T_softmax_maxelem[i0_2])
                    T.writes(T_softmax_exp[i0_2, i1_1])
                    T_softmax_exp[i0_2, i1_1] = T.exp(rxplaceholder[i0_2, i1_1] - T_softmax_maxelem[i0_2], dtype="float32")

            for i0_3, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_softmax_expsum"):
                    i0_4, k = T.axis.remap("SR", [i0_3, i1])
                    T.reads(T_softmax_exp[i0_4, k])
                    T.writes(T_softmax_expsum[i0_4])
                    with T.init():
                        T_softmax_expsum[i0_4] = T.float32(0)
                    T_softmax_expsum[i0_4] = T_softmax_expsum[i0_4] + T_softmax_exp[i0_4, k]

            for i0_5, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_softmax_norm"):
                    i0_6, i1_2 = T.axis.remap("SS", [i0_5, i1])
                    T.reads(T_softmax_exp[i0_6, i1_2], T_softmax_expsum[i0_6])
                    T.writes(T_softmax_norm[i0_6, i1_2])
                    T.block_attr({"axis":1})
                    T_softmax_norm[i0_6, i1_2] = T_softmax_exp[i0_6, i1_2] / T_softmax_expsum[i0_6]

            for i0_7, i1_3 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_8, i1_4 = T.axis.remap("SS", [i0_7, i1_3])
                    T.reads(T_softmax_norm[i0_8, i1_4])
                    T.writes(compute[i0_8, i1_4])
                    compute[i0_8, i1_4] = T.log(T_softmax_norm[i0_8, i1_4], dtype="float32")

            for i0_9, i1_5 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0_9, i1_5])
                    T.reads(compute[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_multiply_1[ax0, ax1])
                    T_multiply_1[ax0, ax1] = compute[ax0, ax1] * rxplaceholder_1[ax0, ax1]

            for i0_10, i1_6 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_red"):
                    k0, k1 = T.axis.remap("RR", [i0_10, i1_6])
                    T.reads(T_multiply_1[k0, k1])
                    T.writes(T_multiply_red[()])
                    with T.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply_1[k0, k1]

            with T.block("T_multiply_1"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_red[()])
                T.writes(T_multiply[()])
                T_multiply[()] = T_multiply_red[()] * T.float32(-1)

    mod = OperatorLegalizer(SoftmaxCrossEntropy).transform()
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
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"],
            rxplaceholder_red: T.Buffer[(T.int64(1), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "sum", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(2), T.int64(4)):
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
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"],
            rxplaceholder_red: T.Buffer[(T.int64(1), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "sum", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(2), T.int64(4)):
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
            rxplaceholder: T.Buffer[(T.int64(1), T.int64(3)), "float32"],
            T_divide: T.Buffer[(T.int64(1), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "divide", "tir.noalias": True})
            for i0, i1 in T.grid(T.int64(1), T.int64(3)):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder[ax0, ax1] * T.float32(0.125)

    mod = OperatorLegalizer(Mean).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


def test_image_resize2d():
    @I.ir_module
    class Resize2D:
        @R.function
        def main(x: R.Tensor((2, 8, 8, 3), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 16, 16, 3), "float32") = R.image.resize2d(
                x,
                size=[16, 16],
                layout="NHWC",
                method="nearest_neighbor",
                coordinate_transformation_mode="asymmetric",
            )
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 8, 8, 3), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv = R.call_tir(resize2d, (x,), (2, 16, 16, 3), dtype="float32")
            return gv

        @T.prim_func
        def resize2d(
            rxplaceholder: T.Buffer[(T.int64(2), T.int64(8), T.int64(8), T.int64(3)), "float32"],
            resize: T.Buffer[(T.int64(2), T.int64(16), T.int64(16), T.int64(3)), "float32"],
        ):
            T.func_attr({"global_symbol": "resize2d", "tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(16), T.int64(16), T.int64(3)):
                with T.block("resize"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(
                        rxplaceholder[
                            i0_1,
                            T.max(T.min(T.Div(i1_1, T.int64(2)), T.int64(7)), T.int64(0)),
                            T.max(T.min(T.Div(i2_1, T.int64(2)), T.int64(7)), T.int64(0)),
                            i3_1,
                        ]
                    )
                    T.writes(resize[i0_1, i1_1, i2_1, i3_1])
                    resize[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[
                        i0_1,
                        T.max(T.min(T.Div(i1_1, T.int64(2)), T.int64(7)), T.int64(0)),
                        T.max(T.min(T.Div(i2_1, T.int64(2)), T.int64(7)), T.int64(0)),
                        i3_1,
                    ]

    mod = OperatorLegalizer(Resize2D).transform()
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    # Todo: test_split_by_indices
    # Todo: test_split_by_n_section
    # Todo: test_batch_norm
    pytest.main([__file__])
