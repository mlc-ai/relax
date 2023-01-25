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
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T
import tvm.testing


def test_customize_legalize_map():
    # fmt: off
    @tvm.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.add(x, y)
            return gv


    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(add, (y, x), (4, 3, 2, 3), dtype="float32")
            return gv

        @T.prim_func
        def add(rxplaceholder_1: T.Buffer[(T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"], rxplaceholder: T.Buffer[(T.int64(1), T.int64(2), T.int64(3)), "float32"], T_add: T.Buffer[(T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32"]):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    T.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] + rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call):
        from tvm import topi  # pylint: disable=import-outside-toplevel

        return bb.call_te(topi.add, call.args[1], call.args[0])

    mod = LegalizeOps({"relax.add": customize_legalize_add})(Add)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_legalize_multiple_types_of_call():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @R.function
        def mul2(x: R.Tensor((3, 3), "float32")):
            gv = R.multiply(x, R.const(2.0, "float32"))
            return gv

        @T.prim_func
        def identity(rxplaceholder: T.Buffer[(T.int64(3), T.int64(3)), "float32"], T_id: T.Buffer[(T.int64(3), T.int64(3)), "float32"]):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_id[v_ax0, v_ax1])
                    T_id[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1]

        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            gv: R.Tensor((3, 3), "float32") = mul2(x)
            gv1 = R.call_tir(identity, gv, (3, 3), dtype="float32")
            gv2 = R.multiply(gv1, R.const(2.0, "float32"))
            return gv2

    @tvm.script.ir_module
    class Expected:
        @R.function
        def mul2(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
            gv = R.call_tir(multiply, (x,), (3, 3), dtype="float32")
            return gv

        @T.prim_func
        def identity(rxplaceholder: T.Buffer[(T.int64(3), T.int64(3)), "float32"], T_id: T.Buffer[(T.int64(3), T.int64(3)), "float32"]):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_id[v_ax0, v_ax1])
                    T_id[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1]

        @T.prim_func
        def multiply(rxplaceholder: T.Buffer[(T.int64(3), T.int64(3)), "float32"], T_multiply: T.Buffer[(T.int64(3), T.int64(3)), "float32"]):
            T.func_attr({"tir.noalias": True})
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1])
                    T.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * T.float32(2)

        @R.function
        def main(x1: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
            gv1: R.Tensor((3, 3), dtype="float32") = mul2(x1)
            gv11 = R.call_tir(identity, gv1, (3, 3), dtype="float32")
            gv2 = R.call_tir(multiply, (gv11,), (3, 3), dtype="float32")
            return gv2
    # fmt: on

    After = LegalizeOps()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
