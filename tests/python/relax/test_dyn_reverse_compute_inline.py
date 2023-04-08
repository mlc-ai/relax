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
import numpy as np
import tvm
import tvm.testing
from tvm import relax, te, tir, meta_schedule as ms
from tvm.script import relax as R, tir as T, ir as I


def test_matmul():
    @T.prim_func
    def main(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(var_rxplaceholder, (n, 4))
        B = T.match_buffer(var_rxplaceholder_1, (4, n))
        C = T.match_buffer(var_matmul, (n, n))
        # with T.block("root"):
        A_pad = T.alloc_buffer(((n + 31) // 32 * 32, 4))
        B_pad = T.alloc_buffer((4, (n + 31) // 32 * 32))
        C_pad = T.alloc_buffer(((n + 31) // 32 * 32, (n + 31) // 32 * 32))
        for i0, i1 in T.grid((n + 31) // 32 * 32, 4):
            with T.block("A_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[v0, v1])
                T.writes(A_pad[v0, v1])
                A_pad[v0, v1] = T.if_then_else(v0 < n, A[v0, v1], T.float32(0))
        for i0, i1 in T.grid(4, (n + 31) // 32 * 32):
            with T.block("B_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                T.reads(B[v0, v1])
                T.writes(B_pad[v0, v1])
                B_pad[v0, v1] = T.if_then_else(v1 < n, B[v0, v1], T.float32(0))
        for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
            with T.block("matmul_o"):
                v_i0_o, v_i1_o = T.axis.remap("SS", [i0_0, i1_0])
                T.reads(
                    A_pad[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                    B_pad[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
                )
                T.writes(C_pad[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
                C_pad_local = T.alloc_buffer((32, 32), scope="local")
                for i0_1, i1_1, k in T.grid(32, 32, 4):
                    with T.block("matmul"):
                        v_i0_i, v_i1_i, v_k_i = T.axis.remap("SSR", [i0_1, i1_1, k])
                        T.reads(
                            A_pad[v_i0_o * 32 + v_i0_i, v_k_i], B_pad[v_k_i, v_i1_o * 32 + v_i1_i]
                        )
                        T.writes(C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i])
                        with T.init():
                            C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = T.float32(0)
                        C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = (
                            C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i]
                            + A_pad[v_i0_o * 32 + v_i0_i, v_k_i]
                            * B_pad[v_k_i, v_i1_o * 32 + v_i1_i]
                        )
                for ax0, ax1 in T.grid(32, 32):
                    with T.block("C_pad_local"):
                        v0 = T.axis.spatial(32, ax0)
                        v1 = T.axis.spatial(32, ax1)
                        T.reads(C_pad_local[v0, v1])
                        T.writes(C_pad[v_i0_o * 32 + v0, v_i1_o * 32 + v1])
                        C_pad[v_i0_o * 32 + v0, v_i1_o * 32 + v1] = C_pad_local[v0, v1]
        for i0, i1 in T.grid(n, n):
            with T.block("C_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                T.reads(C_pad[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_pad[v0, v1]

    @T.prim_func
    def expected(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(var_rxplaceholder, (n, 4))
        B = T.match_buffer(var_rxplaceholder_1, (4, n))
        C = T.match_buffer(var_matmul, (n, n))
        # with T.block("root"):
        A_pad = T.alloc_buffer(((n + 31) // 32 * 32, 4))
        B_pad = T.alloc_buffer((4, (n + 31) // 32 * 32))
        for i0, i1 in T.grid((n + 31) // 32 * 32, 4):
            with T.block("A_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[v0, v1])
                T.writes(A_pad[v0, v1])
                A_pad[v0, v1] = T.if_then_else(v0 < n, A[v0, v1], T.float32(0))
        for i0, i1 in T.grid(4, (n + 31) // 32 * 32):
            with T.block("B_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                T.reads(B[v0, v1])
                T.writes(B_pad[v0, v1])
                B_pad[v0, v1] = T.if_then_else(v1 < n, B[v0, v1], T.float32(0))
        for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
            with T.block("matmul_o"):
                v_i0_o, v_i1_o = T.axis.remap("SS", [i0_0, i1_0])
                T.reads(
                    A_pad[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                    B_pad[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
                )
                T.writes(C[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
                C_pad_local = T.alloc_buffer((32, 32), scope="local")
                for i0_1, i1_1, k in T.grid(32, 32, 4):
                    with T.block("matmul"):
                        v_i0_i, v_i1_i, v_k_i = T.axis.remap("SSR", [i0_1, i1_1, k])
                        T.reads(
                            A_pad[v_i0_o * 32 + v_i0_i, v_k_i], B_pad[v_k_i, v_i1_o * 32 + v_i1_i]
                        )
                        T.writes(C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i])
                        with T.init():
                            C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = T.float32(0)
                        C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = (
                            C_pad_local[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i]
                            + A_pad[v_i0_o * 32 + v_i0_i, v_k_i]
                            * B_pad[v_k_i, v_i1_o * 32 + v_i1_i]
                        )
                for ax0, ax1 in T.grid(32, 32):
                    with T.block("C_pad_local"):
                        v0 = T.axis.spatial(32, ax0)
                        v1 = T.axis.spatial(32, ax1)
                        T.reads(C_pad_local[v0, v1])
                        T.writes(C[v_i0_o * 32 + v0, v_i1_o * 32 + v1])
                        T.where(
                            0 <= v_i0_o * 32 + ax0
                            and v_i0_o * 32 + ax0 < n
                            and 0 <= v_i1_o * 32 + ax1
                            and v_i1_o * 32 + ax1 < n
                        )
                        C[v_i0_o * 32 + v0, v_i1_o * 32 + v1] = C_pad_local[v0, v1]

    sch = tir.Schedule(main, debug_mask="all")
    b0 = sch.get_block(name="matmul", func_name="main")
    b1 = sch.get_block(name="C_pad", func_name="main")
    sch.reverse_compute_inline(b1)
    tvm.ir.assert_structural_equal(sch.mod["main"], expected)


def test_norm_s4():
    @I.ir_module
    class ModAfterS3:
        @T.prim_func
        def main(
            var_A: T.handle,
            var_weight: T.Buffer((T.int64(4096),), "float32"),
            var_rms_norm: T.handle,
        ):
            T.func_attr(
                {"op_pattern": 4, "tir.noalias": T.bool(True), "tir_var_upper_bound": {"n": 2048}}
            )
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)))
            rms_norm = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)))
            sq_sum = T.alloc_buffer((T.int64(1), n))

            sq_sum_pad = T.alloc_buffer(
                [T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32)], dtype="float32"
            )

            # compute on padded buffers
            for i_0 in range((n + T.int64(31)) // T.int64(32)):
                with T.block("compute_o"):
                    v_bsz = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i_0)
                    T.reads(
                        A[
                            v_bsz,
                            v_i_o * T.int64(32) : v_i_o * T.int64(32) + T.int64(32),
                            T.int64(0) : T.int64(4096),
                        ]
                    )
                    T.writes(
                        sq_sum_pad[v_bsz, v_i_o * T.int64(32) : v_i_o * T.int64(32) + T.int64(32)]
                    )
                    sq_sum_pad_local = T.alloc_buffer([T.int64(32)], dtype="float32", scope="local")
                    for bsz, i_1, k in T.grid(T.int64(1), T.int64(32), T.int64(4096)):
                        with T.block("compute"):
                            v_i_i, v_k_i = T.axis.remap("SR", [i_1, k])
                            T.reads(A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i])
                            T.writes(sq_sum_pad_local[v_i_i])
                            with T.init():
                                sq_sum_pad_local[v_i_i] = T.float32(0)
                            sq_sum_pad_local[v_i_i] = sq_sum_pad_local[v_i_i] + T.if_then_else(
                                v_i_o * T.int64(32) + v_i_i < n,
                                A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i],
                                T.float32(0),
                            ) * T.if_then_else(
                                v_i_o * T.int64(32) + v_i_i < n,
                                A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i],
                                T.float32(0),
                            )
                    for bsz, i_1 in T.grid(T.int64(1), T.int64(32)):
                        with T.block("compute_cache_write"):
                            v_i_i = T.axis.remap("S", [i_1])
                            T.reads(sq_sum_pad_local[v_i_i])
                            T.writes(sq_sum_pad[v_bsz, v_i_o * T.int64(32) + v_i_i])
                            sq_sum_pad[v_bsz, v_i_o * T.int64(32) + v_i_i] = sq_sum_pad_local[v_i_i]

            # write back to sq_sum
            for bsz, i in T.grid(T.int64(1), n):
                with T.block("sq_sum_pad"):
                    v_bsz, v_i = T.axis.remap("SS", [bsz, i])
                    T.reads(sq_sum_pad[v_bsz, v_i])
                    T.writes(sq_sum[v_bsz, v_i])
                    sq_sum[v_bsz, v_i] = sq_sum_pad[v_bsz, v_i]

            for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
                with T.block("rms_norm"):
                    v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                    T.reads(var_weight[v_k], A[v_bsz, v_i, v_k], sq_sum[v_bsz, v_i])
                    T.writes(rms_norm[v_bsz, v_i, v_k])
                    rms_norm[v_bsz, v_i, v_k] = var_weight[v_k] * (
                        A[v_bsz, v_i, v_k]
                        / T.sqrt(
                            sq_sum[v_bsz, v_i] * T.float32(0.000244140625)
                            + T.float32(9.9999999999999995e-07)
                        )
                    )

    @I.ir_module
    class ModAfterS4:
        @T.prim_func
        def main(
            var_A: T.handle,
            var_weight: T.Buffer((T.int64(4096),), "float32"),
            var_rms_norm: T.handle,
        ):
            T.func_attr(
                {"op_pattern": 4, "tir.noalias": T.bool(True), "tir_var_upper_bound": {"n": 2048}}
            )
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)))
            rms_norm = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)))
            sq_sum = T.alloc_buffer((T.int64(1), n))

            # compute on padded buffers
            for i_0 in range((n + T.int64(31)) // T.int64(32)):
                with T.block("compute_o"):
                    v_bsz = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i_o = T.axis.spatial((n + T.int64(31)) // T.int64(32), i_0)
                    T.reads(
                        A[
                            v_bsz,
                            v_i_o * T.int64(32) : v_i_o * T.int64(32) + T.int64(32),
                            T.int64(0) : T.int64(4096),
                        ]
                    )
                    T.writes(sq_sum[v_bsz, v_i_o * T.int64(32) : v_i_o * T.int64(32) + T.int64(32)])
                    sq_sum_pad_local = T.alloc_buffer([T.int64(32)], dtype="float32", scope="local")
                    for bsz, i_1, k in T.grid(T.int64(1), T.int64(32), T.int64(4096)):
                        with T.block("compute"):
                            v_i_i, v_k_i = T.axis.remap("SR", [i_1, k])
                            T.reads(A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i])
                            T.writes(sq_sum_pad_local[v_i_i])
                            with T.init():
                                sq_sum_pad_local[v_i_i] = T.float32(0)
                            sq_sum_pad_local[v_i_i] = sq_sum_pad_local[v_i_i] + T.if_then_else(
                                v_i_o * T.int64(32) + v_i_i < n,
                                A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i],
                                T.float32(0),
                            ) * T.if_then_else(
                                v_i_o * T.int64(32) + v_i_i < n,
                                A[v_bsz, v_i_o * T.int64(32) + v_i_i, v_k_i],
                                T.float32(0),
                            )
                    for bsz, i_1 in T.grid(T.int64(1), T.int64(32)):
                        with T.block("compute_cache_write"):
                            v_i_i = T.axis.remap("S", [i_1])
                            T.reads(sq_sum_pad_local[v_i_i])
                            T.writes(sq_sum[v_bsz, v_i_o * T.int64(32) + v_i_i])
                            T.where(
                                T.int64(0) <= v_bsz
                                and v_bsz < T.int64(1)
                                and T.int64(0) <= v_i_o * T.int64(32) + i_1
                                and v_i_o * T.int64(32) + i_1 < n
                            )
                            sq_sum[v_bsz, v_i_o * T.int64(32) + v_i_i] = sq_sum_pad_local[v_i_i]

            for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
                with T.block("rms_norm"):
                    v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                    T.reads(var_weight[v_k], A[v_bsz, v_i, v_k], sq_sum[v_bsz, v_i])
                    T.writes(rms_norm[v_bsz, v_i, v_k])
                    rms_norm[v_bsz, v_i, v_k] = var_weight[v_k] * (
                        A[v_bsz, v_i, v_k]
                        / T.sqrt(
                            sq_sum[v_bsz, v_i] * T.float32(0.000244140625)
                            + T.float32(9.9999999999999995e-07)
                        )
                    )

    sch = tir.Schedule(ModAfterS3, debug_mask="all")
    sch.reverse_compute_inline(sch.get_block("sq_sum_pad"))
    tvm.ir.assert_structural_equal(sch.mod, ModAfterS4)


if __name__ == "__main__":
    test_matmul()
    test_norm_s4()
