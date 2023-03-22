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
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
import numpy as np
import tvm.topi.testing


def test_simple():
    @tvm.script.ir_module
    class Before:
        @R.function
        def func1(
            x: R.Tensor((256, 256), "float32"),
            w1: R.Tensor((256, 256), "float32"),
        ) -> R.Tensor((256, 256), "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                w1_t = R.permute_dims(w1, [1, 0])
                y = R.matmul(x, w1_t)
                R.output(y)
            return y

    @I.ir_module
    class Expected:
        @T.prim_func
        def decode(
            rxplaceholder: T.Buffer((T.int64(256), T.int64(4), T.int64(32)), "uint8"),
            rxplaceholder_1: T.Buffer((T.int64(256), T.int64(4), T.int64(1)), "float32"),
            rxplaceholder_2: T.Buffer((T.int64(256), T.int64(4), T.int64(1)), "float32"),
            T_reshape: T.Buffer((T.int64(256), T.int64(256)), "float32"),
        ):
            T.func_attr({"StopLifting": True, "tir.noalias": True})
            # with T.block("root"):
            T_bitwise_and = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(32)), "uint8")
            T_right_shift = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(32)), "uint8")
            T_concat = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)), "uint8")
            compute = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            T_subtract = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(1)))
            T_multiply = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            T_divide = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            T_add = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(32)):
                with T.block("T_bitwise_and"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2])
                    T.writes(T_bitwise_and[v_ax0, v_ax1, v_ax2])
                    T_bitwise_and[v_ax0, v_ax1, v_ax2] = T.bitwise_and(
                        rxplaceholder[v_ax0, v_ax1, v_ax2], T.uint8(15)
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(32)):
                with T.block("T_right_shift"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2])
                    T.writes(T_right_shift[v_ax0, v_ax1, v_ax2])
                    T_right_shift[v_ax0, v_ax1, v_ax2] = T.shift_right(
                        rxplaceholder[v_ax0, v_ax1, v_ax2], T.uint8(4)
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_concat"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        T_bitwise_and[v_ax0, v_ax1, v_ax2 - T.int64(32)],
                        T_right_shift[v_ax0, v_ax1, v_ax2],
                    )
                    T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                    T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(
                        T.int64(32) <= v_ax2,
                        T_bitwise_and[v_ax0, v_ax1, v_ax2 - T.int64(32)],
                        T_right_shift[v_ax0, v_ax1, v_ax2],
                    )
            for i0, i1, i2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(T_concat[v_i0, v_i1, v_i2])
                    T.writes(compute[v_i0, v_i1, v_i2])
                    compute[v_i0, v_i1, v_i2] = T.Cast("float32", T_concat[v_i0, v_i1, v_i2])
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(1)):
                with T.block("T_subtract"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder_2[v_ax0, v_ax1, v_ax2]
                    )
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2])
                    T_subtract[v_ax0, v_ax1, v_ax2] = (
                        rxplaceholder_1[v_ax0, v_ax1, v_ax2] - rxplaceholder_2[v_ax0, v_ax1, v_ax2]
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(compute[v_ax0, v_ax1, v_ax2], T_subtract[v_ax0, v_ax1, T.int64(0)])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = (
                        compute[v_ax0, v_ax1, v_ax2] * T_subtract[v_ax0, v_ax1, T.int64(0)]
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_divide"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = T_multiply[v_ax0, v_ax1, v_ax2] * T.float32(
                        0.066666666666666666
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        T_divide[v_ax0, v_ax1, v_ax2], rxplaceholder_2[v_ax0, v_ax1, T.int64(0)]
                    )
                    T.writes(T_add[v_ax0, v_ax1, v_ax2])
                    T_add[v_ax0, v_ax1, v_ax2] = (
                        T_divide[v_ax0, v_ax1, v_ax2] + rxplaceholder_2[v_ax0, v_ax1, T.int64(0)]
                    )
            for ax0, ax1 in T.grid(T.int64(256), T.int64(256)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(
                        T_add[
                            (v_ax1 // T.int64(256) + v_ax0) % T.int64(256),
                            v_ax1 % T.int64(256) // T.int64(64),
                            v_ax1 % T.int64(64),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = T_add[
                        (v_ax1 // T.int64(256) + v_ax0) % T.int64(256),
                        v_ax1 % T.int64(256) // T.int64(64),
                        v_ax1 % T.int64(64),
                    ]

        @T.prim_func
        def encode(
            rxplaceholder: T.Buffer((T.int64(256), T.int64(256)), "float32"),
            T_bitwise_or: T.Buffer((T.int64(256), T.int64(4), T.int64(32)), "uint8"),
            T_reshape_red: T.Buffer((T.int64(256), T.int64(4), T.int64(1)), "float32"),
            T_reshape_red_1: T.Buffer((T.int64(256), T.int64(4), T.int64(1)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            T_reshape = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            T_subtract = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            T_multiply = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            T_subtract_1 = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(1)))
            T_divide = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            compute = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)))
            compute_1 = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(64)), "uint8")
            T_strided_slice_with_axes = T.alloc_buffer(
                (T.int64(256), T.int64(4), T.int64(32)), "uint8"
            )
            T_left_shift = T.alloc_buffer((T.int64(256), T.int64(4), T.int64(32)), "uint8")
            T_strided_slice_with_axes_1 = T.alloc_buffer(
                (T.int64(256), T.int64(4), T.int64(32)), "uint8"
            )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        rxplaceholder[
                            ((v_ax1 * T.int64(64) + v_ax2) // T.int64(256) + v_ax0) % T.int64(256),
                            (v_ax1 * T.int64(64) + v_ax2) % T.int64(256),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        ((v_ax1 * T.int64(64) + v_ax2) // T.int64(256) + v_ax0) % T.int64(256),
                        (v_ax1 * T.int64(64) + v_ax2) % T.int64(256),
                    ]
            for ax0, ax1, ax2, k2 in T.grid(T.int64(256), T.int64(4), T.int64(1), T.int64(64)):
                with T.block("T_reshape_red"):
                    v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                    T.reads(T_reshape[v_ax0, v_ax1, v_k2])
                    T.writes(T_reshape_red_1[v_ax0, v_ax1, v_ax2])
                    with T.init():
                        T_reshape_red_1[v_ax0, v_ax1, v_ax2] = T.float32(3.4028234663852886e38)
                    T_reshape_red_1[v_ax0, v_ax1, v_ax2] = T.min(
                        T_reshape_red_1[v_ax0, v_ax1, v_ax2], T_reshape[v_ax0, v_ax1, v_k2]
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_subtract"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        T_reshape[v_ax0, v_ax1, v_ax2], T_reshape_red_1[v_ax0, v_ax1, T.int64(0)]
                    )
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2])
                    T_subtract[v_ax0, v_ax1, v_ax2] = (
                        T_reshape[v_ax0, v_ax1, v_ax2] - T_reshape_red_1[v_ax0, v_ax1, T.int64(0)]
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = (
                        T.float32(15) * T_subtract[v_ax0, v_ax1, v_ax2]
                    )
            for ax0, ax1, ax2, k2 in T.grid(T.int64(256), T.int64(4), T.int64(1), T.int64(64)):
                with T.block("T_reshape_red_1"):
                    v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                    T.reads(T_reshape[v_ax0, v_ax1, v_k2])
                    T.writes(T_reshape_red[v_ax0, v_ax1, v_ax2])
                    with T.init():
                        T_reshape_red[v_ax0, v_ax1, v_ax2] = T.float32(-3.4028234663852886e38)
                    T_reshape_red[v_ax0, v_ax1, v_ax2] = T.max(
                        T_reshape_red[v_ax0, v_ax1, v_ax2], T_reshape[v_ax0, v_ax1, v_k2]
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(1)):
                with T.block("T_subtract_1"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        T_reshape_red[v_ax0, v_ax1, v_ax2], T_reshape_red_1[v_ax0, v_ax1, v_ax2]
                    )
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2])
                    T_subtract_1[v_ax0, v_ax1, v_ax2] = (
                        T_reshape_red[v_ax0, v_ax1, v_ax2] - T_reshape_red_1[v_ax0, v_ax1, v_ax2]
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("T_divide"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2], T_subtract_1[v_ax0, v_ax1, T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = (
                        T_multiply[v_ax0, v_ax1, v_ax2] / T_subtract_1[v_ax0, v_ax1, T.int64(0)]
                    )
            for i0, i1, i2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(T_divide[v_i0, v_i1, v_i2])
                    T.writes(compute[v_i0, v_i1, v_i2])
                    compute[v_i0, v_i1, v_i2] = T.round(T_divide[v_i0, v_i1, v_i2])
            for i0, i1, i2 in T.grid(T.int64(256), T.int64(4), T.int64(64)):
                with T.block("compute_1"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(compute[v_i0, v_i1, v_i2])
                    T.writes(compute_1[v_i0, v_i1, v_i2])
                    compute_1[v_i0, v_i1, v_i2] = T.Cast("uint8", compute[v_i0, v_i1, v_i2])
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(32)):
                with T.block("T_strided_slice_with_axes"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(compute_1[v_ax0, v_ax1, v_ax2])
                    T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                    T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] = compute_1[v_ax0, v_ax1, v_ax2]
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(32)):
                with T.block("T_left_shift"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                    T.writes(T_left_shift[v_ax0, v_ax1, v_ax2])
                    T_left_shift[v_ax0, v_ax1, v_ax2] = T.shift_left(
                        T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2], T.uint8(4)
                    )
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(32)):
                with T.block("T_strided_slice_with_axes_1"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(compute_1[v_ax0, v_ax1, v_ax2 + T.int64(32)])
                    T.writes(T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2])
                    T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2] = compute_1[
                        v_ax0, v_ax1, v_ax2 + T.int64(32)
                    ]
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(4), T.int64(32)):
                with T.block("T_bitwise_or"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        T_left_shift[v_ax0, v_ax1, v_ax2],
                        T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2],
                    )
                    T.writes(T_bitwise_or[v_ax0, v_ax1, v_ax2])
                    T_bitwise_or[v_ax0, v_ax1, v_ax2] = T.bitwise_or(
                        T_left_shift[v_ax0, v_ax1, v_ax2],
                        T_strided_slice_with_axes_1[v_ax0, v_ax1, v_ax2],
                    )

        @R.function
        def func1(
            x: R.Tensor((256, 256), dtype="float32"), w1: R.Tensor((256, 256), dtype="float32")
        ) -> R.Tensor((256, 256), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (w1,),
                    out_sinfo=[
                        R.Tensor((256, 4, 32), dtype="uint8"),
                        R.Tensor((256, 4, 1), dtype="float32"),
                        R.Tensor((256, 4, 1), dtype="float32"),
                    ],
                )
                lv1: R.Tensor((256, 4, 32), dtype="uint8") = lv[0]
                lv2: R.Tensor((256, 4, 1), dtype="float32") = lv[1]
                lv3: R.Tensor((256, 4, 1), dtype="float32") = lv[2]
                lv4 = R.call_tir(
                    cls.decode, (lv1, lv2, lv3), out_sinfo=R.Tensor((256, 256), dtype="float32")
                )
                w1_t: R.Tensor((256, 256), dtype="float32") = R.permute_dims(lv4, axes=[1, 0])
                y: R.Tensor((256, 256), dtype="float32") = R.matmul(x, w1_t, out_dtype="void")
                R.output(y)
            return y

    mod = Before
    after = relax.transform.GroupQuantize()(Before)
    tvm.ir.assert_structural_equal(after, Expected)

    mod = relax.transform.LegalizeOps()(mod)
    ex_before = relax.build(mod, target="llvm")
    vm_before = relax.VirtualMachine(ex_before, tvm.cpu())
    data_nd = tvm.nd.array(np.random.rand(256, 256).astype("float32"))
    weight_nd = tvm.nd.array(np.random.rand(256, 256).astype("float32"))
    res_before = vm_before["func1"](data_nd, weight_nd)

    after = relax.transform.LegalizeOps()(after)
    ex_after = relax.build(after, target="llvm")
    vm_after = relax.VirtualMachine(ex_after, tvm.cpu())
    res_after = vm_after["func1"](data_nd, weight_nd)

    tvm.testing.assert_allclose(res_before.numpy(), res_after.numpy(), rtol=0.05)


if __name__ == "__main__":
    test_simple()
