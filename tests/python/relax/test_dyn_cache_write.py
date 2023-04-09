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
            for i0_1, i1_1, k in T.grid(32, 32, 4):
                with T.block("matmul"):
                    v_i0_i, v_i1_i, v_k_i = T.axis.remap("SSR", [i0_1, i1_1, k])
                    T.reads(A_pad[v_i0_o * 32 + v_i0_i, v_k_i], B_pad[v_k_i, v_i1_o * 32 + v_i1_i])
                    T.writes(C_pad[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i])
                    with T.init():
                        C_pad[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = T.float32(0)
                    C_pad[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = (
                        C_pad[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i]
                        + A_pad[v_i0_o * 32 + v_i0_i, v_k_i] * B_pad[v_k_i, v_i1_o * 32 + v_i1_i]
                    )
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
                    T.reads(A_pad[v_i0_o * 32 + v_i0_i, v_k_i], B_pad[v_k_i, v_i1_o * 32 + v_i1_i])
                    T.writes(C_pad_local[v_i0_i, v_i1_i])
                    with T.init():
                        C_pad_local[v_i0_i, v_i1_i] = T.float32(0)
                    C_pad_local[v_i0_i, v_i1_i] = (
                        C_pad_local[v_i0_i, v_i1_i]
                        + A_pad[v_i0_o * 32 + v_i0_i, v_k_i] * B_pad[v_k_i, v_i1_o * 32 + v_i1_i]
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


def test():
    sch = tir.Schedule(main, debug_mask="all")
    b0 = sch.get_block(name="matmul", func_name="main")
    b34 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    tvm.ir.assert_structural_equal(sch.mod["main"], expected)


if __name__ == "__main__":
    test()
