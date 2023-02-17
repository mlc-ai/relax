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
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T, ir as I
import tvm.testing


##################### Search #####################


def test_where():
    # fmt: off
    @tvm.script.ir_module
    class Where:
        @R.function
        def main(condition: R.Tensor((3, 2, 1), "bool"), x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")) -> R.Tensor((3, 2, 3), "float32"):
            gv: R.Tensor((3, 2, 3), "float32") = R.where(condition, x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(condition: R.Tensor((3, 2, 1), "bool"), x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")) -> R.Tensor((3, 2, 3), "float32"):
            gv = R.call_tir(where, (condition, x, y), R.Tensor((3, 2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def where(rxplaceholder: T.Buffer[(T.int64(3), T.int64(2), T.int64(1)), "bool"], rxplaceholder_1: T.Buffer[(T.int64(2), T.int64(3)), "float32"], rxplaceholder_2: T.Buffer[(T.int64(2), T.int64(1)), "float32"], T_where: T.Buffer[(T.int64(3), T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_where"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1, T.int64(0)], rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
                    T.writes(T_where[ax0, ax1, ax2])
                    T_where[ax0, ax1, ax2] = T.Select(0 < T.Cast("int32", rxplaceholder[ax0, ax1, T.int64(0)]), rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Where)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_where_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Where:
        @R.function
        def main(condition: R.Tensor(("a", "b", 1), "bool"), x: R.Tensor(("b", "c"), "float32"), y: R.Tensor(("b", 1), "float32")) -> R.Tensor(("a", "b", "c"), "float32"):
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            gv: R.Tensor((a, b, c), "float32") = R.where(condition, x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(condition: R.Tensor(("a", "b", 1), "bool"), x: R.Tensor(("b", "c"), "float32"), y: R.Tensor(("b", 1), "float32")) -> R.Tensor(("a", "b", "c"), "float32"):
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            gv = R.call_tir(where, (condition, x, y), R.Tensor((a, b, c), dtype="float32"))
            return gv

        @T.prim_func
        def where(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_T_where: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, T.int64(1)], dtype="bool")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [b, c], dtype="float32")
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, [b, T.int64(1)], dtype="float32")
            T_where = T.match_buffer(var_T_where, [a, b, c], dtype="float32")
            for i0, i1, i2 in T.grid(a, b, c):
                with T.block("T_where"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1, T.int64(0)], rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
                    T.writes(T_where[ax0, ax1, ax2])
                    T_where[ax0, ax1, ax2] = T.Select(0 < T.Cast("int32", rxplaceholder[ax0, ax1, T.int64(0)]), rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Where)
    tvm.ir.assert_structural_equal(mod, Expected)


##################### Statistical #####################


def test_max():
    # fmt: off
    @tvm.script.ir_module
    class Max:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 5), "float32"):
            gv: R.Tensor((2, 5), "float32") = R.max(x, axis=[1, 2])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 5), "float32"):
            gv = R.call_tir(max, (x,), R.Tensor((2, 5), dtype="float32"))
            return gv

        @T.prim_func
        def max(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"], rxplaceholder_red: T.Buffer[(T.int64(2), T.int64(5)), "float32"]):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(5), T.int64(3), T.int64(4)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k1, k2 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, k1, k2, ax1])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.min_value("float32")
                    rxplaceholder_red[ax0, ax1] = T.max(rxplaceholder_red[ax0, ax1], rxplaceholder[ax0, k1, k2, ax1])
    # fmt: on

    mod = LegalizeOps()(Max)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_max_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Max:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor(("a", "d"), "float32"):
            a = T.var("int64")
            d = T.var("int64")
            gv: R.Tensor((a, d), "float32") = R.max(x, axis=[1, 2])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor(("a", "d"), "float32"):
            a = T.var("int64")
            d = T.var("int64")
            gv = R.call_tir(max, (x,), R.Tensor((a, d), dtype="float32"))
            return gv

        @T.prim_func
        def max(var_rxplaceholder: T.handle, var_rxplaceholder_red: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            d = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c, d], dtype="float32")
            rxplaceholder_red = T.match_buffer(var_rxplaceholder_red, [a, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, d, b, c):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k1, k2 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, k1, k2, ax1])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.min_value("float32")
                    rxplaceholder_red[ax0, ax1] = T.max(rxplaceholder_red[ax0, ax1], rxplaceholder[ax0, k1, k2, ax1])
    # fmt: on

    mod = LegalizeOps()(Max)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_min():
    # fmt: off
    @tvm.script.ir_module
    class Min:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 1, 1, 5), "float32"):
            gv: R.Tensor((2, 1, 1, 5), "float32") = R.min(x, axis=[1, 2], keepdims=True)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 1, 1, 5), "float32"):
            gv = R.call_tir(min, (x,), R.Tensor((2, 1, 1, 5), dtype="float32"))
            return gv

        @T.prim_func
        def min(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"], rxplaceholder_red: T.Buffer[(T.int64(2), T.int64(1), T.int64(1), T.int64(5)), "float32"]):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(5), T.int64(3), T.int64(4)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k1, k2 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[ax0, k1, k2, ax3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.max_value("float32")
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = T.min(rxplaceholder_red[ax0, ax1, ax2, ax3], rxplaceholder[ax0, k1, k2, ax3])
    # fmt: on

    mod = LegalizeOps()(Min)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_min_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Min:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor(("a", 1, 1, "d"), "float32"):
            a = T.var("int64")
            d = T.var("int64")
            gv: R.Tensor((a, 1, 1, d), "float32") = R.min(x, axis=[1, 2], keepdims=True)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor(("a", 1, 1, "d"), "float32"):
            a = T.var("int64")
            d = T.var("int64")
            gv = R.call_tir(min, (x,), R.Tensor((a, 1, 1, d), dtype="float32"))
            return gv

        @T.prim_func
        def min(var_rxplaceholder: T.handle, var_rxplaceholder_red: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            d = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c, d], dtype="float32")
            rxplaceholder_red = T.match_buffer(var_rxplaceholder_red, [a, T.int64(1), T.int64(1), d], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(a, T.int64(1), T.int64(1), d, b, c):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k1, k2 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[ax0, k1, k2, ax3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.max_value("float32")
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = T.min(rxplaceholder_red[ax0, ax1, ax2, ax3], rxplaceholder[ax0, k1, k2, ax3])
    # fmt: on

    mod = LegalizeOps()(Min)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sum():
    # fmt: off
    @tvm.script.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.sum(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "float32"):
            gv = R.call_tir(_te_sum, (x,), out_sinfo=R.Tensor((), "float32"))
            return gv

        @T.prim_func
        def _te_sum(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_red: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            for k0, k1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("rxplaceholder_red"):
                    v_k0, v_k1, v_k2, v_k3 = T.axis.remap("RRRR", [k0, k1, k2, k3])
                    T.reads(rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    T.writes(rxplaceholder_red[()])
                    with T.init():
                        rxplaceholder_red[()] = T.float32(0)
                    rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[v_k0, v_k1, v_k2, v_k3]
    # fmt: on

    mod = LegalizeOps()(Sum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sum_0_dim():
    # fmt: off
    @tvm.script.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor((), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.sum(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((), dtype="float32"):
            gv = R.call_tir(_te_sum, (x,), out_sinfo=R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func
        def _te_sum(rxplaceholder: T.Buffer((), "float32"), compute: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            with T.block("compute"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(rxplaceholder[()])
                T.writes(compute[()])
                compute[()] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(Sum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_sum_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.sum(x)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), dtype="float32")) -> R.Tensor((), dtype="float32"):
            a = T.Var("a", "int64")
            b = T.Var("b", "int64")
            c = T.Var("c", "int64")
            d = T.Var("d", "int64")
            gv = R.call_tir(_te_sum, (x,), out_sinfo=R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func
        def _te_sum(var_rxplaceholder: T.handle, rxplaceholder_red: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            d = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, (a, b, c, d))
            # with T.block("root"):
            for k0, k1, k2, k3 in T.grid(a, b, c, d):
                with T.block("rxplaceholder_red"):
                    v_k0, v_k1, v_k2, v_k3 = T.axis.remap("RRRR", [k0, k1, k2, k3])
                    T.reads(rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    T.writes(rxplaceholder_red[()])
                    with T.init():
                        rxplaceholder_red[()] = T.float32(0)
                    rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[v_k0, v_k1, v_k2, v_k3]
    # fmt: on

    mod = LegalizeOps()(Sum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_prod():
    # fmt: off
    @tvm.script.ir_module
    class Prod:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv: R.Tensor((1, 1, 1, 1), "float32") = R.prod(x, keepdims=True)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv = R.call_tir(prod, (x,), R.Tensor((1, 1, 1, 1), dtype="float32"))
            return gv

        @T.prim_func
        def prod(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"], rxplaceholder_red: T.Buffer[(T.int64(1), T.int64(1), T.int64(1), T.int64(1)), "float32"]):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k1, k2, k3 = T.axis.remap("SSSSRRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                    T.reads(rxplaceholder[k0, k1, k2, k3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(1)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] * rxplaceholder[k0, k1, k2, k3]
    # fmt: on

    mod = LegalizeOps()(Prod)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_prod_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Prod:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv: R.Tensor((1, 1, 1, 1), "float32") = R.prod(x, keepdims=True)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv = R.call_tir(prod, (x,), R.Tensor((1, 1, 1, 1), dtype="float32"))
            return gv

        @T.prim_func
        def prod(var_rxplaceholder: T.handle, rxplaceholder_red: T.Buffer[(T.int64(1), T.int64(1), T.int64(1), T.int64(1)), "float32"]):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            d = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), a, b, c, d):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k1, k2, k3 = T.axis.remap("SSSSRRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                    T.reads(rxplaceholder[k0, k1, k2, k3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(1)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] * rxplaceholder[k0, k1, k2, k3]
    # fmt: on

    mod = LegalizeOps()(Prod)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_mean():
    # fmt: off
    @tvm.script.ir_module
    class Mean:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((3, 4), "float32"):
            gv: R.Tensor((3, 4), "float32") = R.mean(x, [0, 3])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((3, 4), "float32"):
            gv = R.call_tir(mean, (x,), R.Tensor((3, 4), dtype="float32"))
            return gv

        @T.prim_func
        def mean(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"], T_divide: T.Buffer[(T.int64(3), T.int64(4)), "float32"]):
            T.func_attr({"tir.noalias": True})
            rxplaceholder_red = T.alloc_buffer([T.int64(3), T.int64(4)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(3), T.int64(4), T.int64(2), T.int64(5)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k0, k3 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[k0, ax0, ax1, k3])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax0, ax1, k3]
            for i0, i1 in T.grid(T.int64(3), T.int64(4)):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_red[ax0, ax1])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder_red[ax0, ax1] * T.float32(0.1)
    # fmt: on

    mod = LegalizeOps()(Mean)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_mean_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Mean:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor(("b", "c"), "float32"):
            b = T.var("int64")
            c = T.var("int64")
            gv: R.Tensor((b, c), "float32") = R.mean(x, [0, 3])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), dtype="float32")) -> R.Tensor(("b", "c"), dtype="float32"):
            b = T.var("int64")
            c = T.var("int64")
            gv = R.call_tir(mean, (x,), R.Tensor((b, c), dtype="float32"))
            return gv

        @T.prim_func
        def mean(var_rxplaceholder: T.handle, var_T_divide: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            d = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c, d], dtype="float32")
            T_divide = T.match_buffer(var_T_divide, [b, c], dtype="float32")
            rxplaceholder_red = T.alloc_buffer([b, c], dtype="float32")
            for i0, i1, i2, i3 in T.grid(b, c, a, d):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k0, k3 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[k0, ax0, ax1, k3])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax0, ax1, k3]
            for i0, i1 in T.grid(b, c):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_red[ax0, ax1])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder_red[ax0, ax1] / T.Cast("float32", a * d)
    # fmt: on

    mod = LegalizeOps()(Mean)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_std():
    # fmt: off
    @tvm.script.ir_module
    class Std:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.std(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "float32"):
            gv = R.call_tir(std, (x,), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func
        def std(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"], compute: T.Buffer[(), "float32"]):
            T.func_attr({"tir.noalias": True})
            rxplaceholder_red = T.alloc_buffer([], dtype="float32")
            T_divide = T.alloc_buffer([], dtype="float32")
            T_subtract = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32")
            T_multiply = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32")
            T_multiply_red = T.alloc_buffer([], dtype="float32")
            T_divide_1 = T.alloc_buffer([], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("rxplaceholder_red"):
                    k0, k1, k2, k3 = T.axis.remap("RRRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[k0, k1, k2, k3])
                    T.writes(rxplaceholder_red[()])
                    with T.init():
                        rxplaceholder_red[()] = T.float32(0)
                    rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[k0, k1, k2, k3]
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(rxplaceholder_red[()])
                T.writes(T_divide[()])
                T_divide[()] = rxplaceholder_red[()] * T.float32(0.0083333333333333332)
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide[()])
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, ax1, ax2, ax3] - T_divide[()]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_subtract[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] * T_subtract[ax0, ax1, ax2, ax3]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply_red"):
                    k0, k1, k2, k3 = T.axis.remap("RRRR", [i0, i1, i2, i3])
                    T.reads(T_multiply[k0, k1, k2, k3])
                    T.writes(T_multiply_red[()])
                    with T.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[k0, k1, k2, k3]
            with T.block("T_divide_1"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_red[()])
                T.writes(T_divide_1[()])
                T_divide_1[()] = T_multiply_red[()] * T.float32(0.0083333333333333332)
            with T.block("compute"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_divide_1[()])
                T.writes(compute[()])
                compute[()] = T.sqrt(T_divide_1[()])
    # fmt: on

    mod = LegalizeOps()(Std)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_std_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Std:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.std(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor((), "float32"):
            gv = R.call_tir(std, (x,), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func
        def std(var_rxplaceholder: T.handle, compute: T.Buffer[(), "float32"]):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            d = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c, d], dtype="float32")
            rxplaceholder_red = T.alloc_buffer([], dtype="float32")
            T_divide = T.alloc_buffer([], dtype="float32")
            T_subtract = T.alloc_buffer([a, b, c, d], dtype="float32")
            T_multiply = T.alloc_buffer([a, b, c, d], dtype="float32")
            T_multiply_red = T.alloc_buffer([], dtype="float32")
            T_divide_1 = T.alloc_buffer([], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("rxplaceholder_red"):
                    k0, k1, k2, k3 = T.axis.remap("RRRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[k0, k1, k2, k3])
                    T.writes(rxplaceholder_red[()])
                    with T.init():
                        rxplaceholder_red[()] = T.float32(0)
                    rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[k0, k1, k2, k3]
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(rxplaceholder_red[()])
                T.writes(T_divide[()])
                T_divide[()] = rxplaceholder_red[()] / T.Cast("float32", a * b * c * d)
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide[()])
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, ax1, ax2, ax3] - T_divide[()]
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_subtract[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] * T_subtract[ax0, ax1, ax2, ax3]
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_multiply_red"):
                    k0, k1, k2, k3 = T.axis.remap("RRRR", [i0, i1, i2, i3])
                    T.reads(T_multiply[k0, k1, k2, k3])
                    T.writes(T_multiply_red[()])
                    with T.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[k0, k1, k2, k3]
            with T.block("T_divide_1"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_red[()])
                T.writes(T_divide_1[()])
                T_divide_1[()] = T_multiply_red[()] / T.Cast("float32", a * b * c * d)
            with T.block("compute"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_divide_1[()])
                T.writes(compute[()])
                compute[()] = T.sqrt(T_divide_1[()])
    # fmt: on

    mod = LegalizeOps()(Std)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_variance():
    # fmt: off
    @tvm.script.ir_module
    class Variance:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((1, 3, 4, 1), "float32"):
            gv: R.Tensor((1, 3, 4, 1), "float32") = R.variance(x, [0, 3], keepdims=True)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((1, 3, 4, 1), dtype="float32"):
            gv = R.call_tir(variance, (x,), R.Tensor((1, 3, 4, 1), dtype="float32"))
            return gv

        @T.prim_func
        def variance(rxplaceholder: T.Buffer[(T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"], T_divide: T.Buffer[(T.int64(1), T.int64(3), T.int64(4), T.int64(1)), "float32"]):
            T.func_attr({"tir.noalias": True})
            rxplaceholder_red = T.alloc_buffer([T.int64(1), T.int64(3), T.int64(4), T.int64(1)], dtype="float32")
            T_divide_1 = T.alloc_buffer([T.int64(1), T.int64(3), T.int64(4), T.int64(1)], dtype="float32")
            T_subtract = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32")
            T_multiply = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32")
            T_multiply_red = T.alloc_buffer([T.int64(1), T.int64(3), T.int64(4), T.int64(1)], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1), T.int64(2), T.int64(5)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[k0, ax1, ax2, k3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] + rxplaceholder[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1)):
                with T.block("T_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide_1[ax0, ax1, ax2, ax3])
                    T_divide_1[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] * T.float32(0.10000000000000001)
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide_1[T.int64(0), ax1, ax2, T.int64(0)])
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, ax1, ax2, ax3] - T_divide_1[T.int64(0), ax1, ax2, T.int64(0)]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_subtract[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] * T_subtract[ax0, ax1, ax2, ax3]
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1), T.int64(2), T.int64(5)):
                with T.block("T_multiply_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(T_multiply[k0, ax1, ax2, k3])
                    T.writes(T_multiply_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        T_multiply_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    T_multiply_red[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] + T_multiply[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1)):
                with T.block("T_divide_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_multiply_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] * T.float32(0.10000000000000001)
    # fmt: on

    mod = LegalizeOps()(Variance)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_variance_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Variance:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor((1, "b", "c", 1), "float32"):
            b = T.var("int64")
            c = T.var("int64")
            gv: R.Tensor((1, b, c, 1), "float32") = R.variance(x, [0, 3], keepdims=True)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor((1, "b", "c", 1), "float32"):
            b = T.var("int64")
            c = T.var("int64")
            gv = R.call_tir(variance, (x,), R.Tensor((1, b, c, 1), dtype="float32"))
            return gv

        @T.prim_func
        def variance(var_rxplaceholder: T.handle, var_T_divide: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.var("int64")
            b = T.var("int64")
            c = T.var("int64")
            d = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c, d], dtype="float32")
            T_divide = T.match_buffer(var_T_divide, [T.int64(1), b, c, T.int64(1)], dtype="float32")
            rxplaceholder_red = T.alloc_buffer([T.int64(1), b, c, T.int64(1)], dtype="float32")
            T_divide_1 = T.alloc_buffer([T.int64(1), b, c, T.int64(1)], dtype="float32")
            T_subtract = T.alloc_buffer([a, b, c, d], dtype="float32")
            T_multiply = T.alloc_buffer([a, b, c, d], dtype="float32")
            T_multiply_red = T.alloc_buffer([T.int64(1), b, c, T.int64(1)], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), b, c, T.int64(1), a, d):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[k0, ax1, ax2, k3])
                    T.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] + rxplaceholder[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), b, c, T.int64(1)):
                with T.block("T_divide"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide_1[ax0, ax1, ax2, ax3])
                    T_divide_1[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] / T.Cast("float32", a * d)
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_subtract"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide_1[T.int64(0), ax1, ax2, T.int64(0)])
                    T.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, ax1, ax2, ax3] - T_divide_1[T.int64(0), ax1, ax2, T.int64(0)]
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_multiply"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_subtract[ax0, ax1, ax2, ax3])
                    T.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] * T_subtract[ax0, ax1, ax2, ax3]
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), b, c, T.int64(1), a, d):
                with T.block("T_multiply_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(T_multiply[k0, ax1, ax2, k3])
                    T.writes(T_multiply_red[ax0, ax1, ax2, ax3])
                    with T.init():
                        T_multiply_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    T_multiply_red[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] + T_multiply[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), b, c, T.int64(1)):
                with T.block("T_divide_1"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_multiply_red[ax0, ax1, ax2, ax3])
                    T.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] / T.Cast("float32", a * d)
    # fmt: on

    mod = LegalizeOps()(Variance)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
