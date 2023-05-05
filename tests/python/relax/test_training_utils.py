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
"""Unit tests for relax training utils."""
import tvm
import tvm.testing
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import relax as R, tir as T, ir as I

from tvm.relax.training.utils import bind_te_grad_func
from tvm.relax.transform import Gradient


def test_bind_tir_grad_func():
    @I.ir_module
    class Expected:
        @T.prim_func
        def f_mul(
            A: T.Buffer((T.int64(5), T.int64(5)), "float32"),
            B: T.Buffer((T.int64(5), T.int64(5)), "float32"),
            f_mul_1: T.Buffer((T.int64(5), T.int64(5)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i0, v_i1])
                    T.writes(f_mul_1[v_i0, v_i1])
                    f_mul_1[v_i0, v_i1] = A[v_i0, v_i1] * B[v_i0, v_i1]

        @T.prim_func
        def f_mul_grad(
            A: T.Buffer((T.int64(5), T.int64(5)), "float32"),
            B: T.Buffer((T.int64(5), T.int64(5)), "float32"),
            C: T.Buffer((T.int64(5), T.int64(5)), "float32"),
            f_mul_grad_1: T.Buffer((T.int64(5), T.int64(5)), "float32"),
            f_mul_grad_2: T.Buffer((T.int64(5), T.int64(5)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul_grad_1"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(C[v_i0, v_i1], A[v_i0, v_i1])
                    T.writes(f_mul_grad_1[v_i0, v_i1])
                    f_mul_grad_1[v_i0, v_i1] = C[v_i0, v_i1] * A[v_i0, v_i1]
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with T.block("f_mul_grad_2"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(B[v_i0, v_i1], A[v_i0, v_i1])
                    T.writes(f_mul_grad_2[v_i0, v_i1])
                    f_mul_grad_2[v_i0, v_i1] = B[v_i0, v_i1] * A[v_i0, v_i1]

        @R.function
        def main_adjoint(
            a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")
        ) -> R.Tuple(
            R.Tensor((), dtype="float32"),
            R.Tuple(R.Tensor((5, 5), dtype="float32"), R.Tensor((5, 5), dtype="float32")),
        ):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a, b), out_sinfo=R.Tensor((5, 5), dtype="float32"))
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(
                    gv_adjoint, R.shape([5, 5])
                )
                lv_1 = R.call_tir(
                    cls.f_mul_grad,
                    (lv_adjoint, a, b),
                    out_sinfo=[
                        R.Tensor((5, 5), dtype="float32"),
                        R.Tensor((5, 5), dtype="float32"),
                    ],
                )
                a_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1[0]
                b_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1[1]
                R.output(gv, a_adjoint, b_adjoint)
            return (gv, (a_adjoint, b_adjoint))

        @R.function
        def main(
            a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")
        ) -> R.Tensor((), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a, b), out_sinfo=R.Tensor((5, 5), dtype="float32"))
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv

    def f_mul(src1, src2):
        def mul(*idx):
            return src1[idx] * src2[idx]

        return tvm.te.compute(src1.shape, mul, name="f_mul")

    def f_mul_grad(output_grad, src1, src2):
        def mul_grad_1(*idx):
            return src2[idx] * output_grad[idx]

        def mul_grad_2(*idx):
            return src1[idx] * output_grad[idx]

        return [
            tvm.te.compute(output_grad.shape, mul_grad_1, name="f_mul_grad_1"),
            tvm.te.compute(output_grad.shape, mul_grad_2, name="f_mul_grad_2"),
        ]

    a = relax.Var("a", relax.TensorStructInfo([5, 5], "float32"))
    b = relax.Var("b", relax.TensorStructInfo([5, 5], "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [a, b]):
        with bb.dataflow():
            d = bb.emit_te(f_mul, a, b, primfunc_name_hint="f_mul")
            out = bb.emit_output(R.sum(d))
        bb.emit_func_output(out)

    mod = bb.get()
    mod = bind_te_grad_func(mod, "f_mul", f_mul_grad)
    with tvm.transform.PassContext():
        After = Gradient("main")(mod)

    # remove the module attr to pass the assert structrual equal
    After = After.without_attr("te_grad_bind_handler")

    assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
