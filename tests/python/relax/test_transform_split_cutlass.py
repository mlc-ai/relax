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

from __future__ import annotations
from tvm.script import relax as R, tir as T
import pytest
from tvm.relay.expr import const
import tvm
import tvm.testing
from tvm import relax
import tvm.script
import numpy as np
from tvm.tir import StringImm

import tvm.relax.cutlass.pattern

GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


@tvm.script.ir_module
class GEMM:
    @T.prim_func
    def HGEMM(
        A: T.Buffer[(16, 32), "float16"],
        B: T.Buffer[(32, 64), "float16"],
        C: T.Buffer[(16, 64), "float16"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "HGEMM", "cutlass_codegen": 1})
        # body
        # with T.block("root")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads(A[v_3, v_5], B[v_5, v_4])
                T.writes(C[v_3, v_4])
                with T.init():
                    C[v_3, v_4] = T.float16(0)
                C[v_3, v_4] = C[v_3, v_4] + A[v_3, v_5] * B[v_5, v_4]

    @R.function
    def main(
        A: R.Tensor((16, 32), "float16"), B: R.Tensor((32, 64), "float16")
    ) -> R.Tensor(None, "float16", ndim=2):
        return R.call_tir(HGEMM, (A, B), (16, 64), dtype="float16")


@tvm.script.ir_module
class GEMM_ReLU:
    @T.prim_func
    def HGEMM(
        A: T.Buffer[(16, 32), "float16"],
        B: T.Buffer[(32, 64), "float16"],
        C: T.Buffer[(16, 64), "float16"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "HGEMM", "cutlass_codegen": 1})
        # body
        # with T.block("root")
        D = T.alloc_buffer([16, 64], dtype="float16")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads(A[v_3, v_5], B[v_5, v_4])
                T.writes(D[v_3, v_4])
                with T.init():
                    D[v_3, v_4] = T.float16(0)
                D[v_3, v_4] = D[v_3, v_4] + A[v_3, v_5] * B[v_5, v_4]
        for v_6, v_7 in T.grid(16, 64):
            with T.block("root"):
                v_8, v_9 = T.axis.remap("SS", [v_6, v_7])
                T.reads(D[v_8, v_9])
                T.writes(C[v_8, v_9])
                C[v_8, v_9] = T.max(D[v_8, v_9], T.float16(0))

    @R.function
    def main(
        A: R.Tensor((16, 32), "float16"), B: R.Tensor((32, 64), "float16")
    ) -> R.Tensor((16, 64), "float16", ndim=2):
        return R.call_tir(HGEMM, (A, B), (16, 64), dtype="float16")


@tvm.script.ir_module
class GEMM_bias_relu_exp:
    @T.prim_func
    def HGEMM(
        A: T.Buffer[(16, 32), "float16"],
        B: T.Buffer[(32, 64), "float16"],
        Bias: T.Buffer[(1, 16), "float16"],
        C: T.Buffer[(16, 64), "float16"],
    ) -> None:
        # function attr dict
        T.func_attr({"cutlass_codegen": 1})
        # body
        # with T.block("root")
        buf_ = T.alloc_buffer([16, 64], dtype="float16")
        buf__1 = T.alloc_buffer([16, 64], dtype="float16")
        buf__2 = T.alloc_buffer([16, 64], dtype="float16")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads(A[v_3, v_5], B[v_5, v_4])
                T.writes(buf_[v_3, v_4])
                with T.init():
                    buf_[v_3, v_4] = T.float16(0)
                buf_[v_3, v_4] = buf_[v_3, v_4] + A[v_3, v_5] * B[v_5, v_4]
        for v_6, v_7 in T.grid(16, 64):
            with T.block("bias"):
                v_8, v_9 = T.axis.remap("SS", [v_6, v_7])
                T.reads(buf_[v_8, v_9], Bias[0, v_9])
                T.writes(buf__1[v_8, v_9])
                buf__1[v_8, v_9] = buf_[v_8, v_9] + Bias[0, v_9]
        for v_10, v_11 in T.grid(16, 64):
            with T.block("relu"):
                v_12, v_13 = T.axis.remap("SS", [v_10, v_11])
                T.reads(buf__1[v_12, v_13])
                T.writes(buf__2[v_12, v_13])
                buf__2[v_12, v_13] = T.max(buf__1[v_12, v_13], T.float16(0))
        for v_14, v_15 in T.grid(16, 64):
            with T.block("exp"):
                v_16, v_17 = T.axis.remap("SS", [v_14, v_15])
                T.reads(buf__2[v_16, v_17])
                T.writes(C[v_16, v_17])
                C[v_16, v_17] = T.exp(buf__2[v_16, v_17], dtype="float16")

    @R.function
    def main(
        A: R.Tensor((16, 32), "float16"),
        B: R.Tensor((32, 64), "float16"),
        Bias: R.Tensor((1, 16), "float16"),
    ) -> R.Tensor(None, "float16", ndim=2):
        return R.call_tir(HGEMM, (A, B, Bias), (16, 64), dtype="float16")


@tvm.script.ir_module
class test_cutlass_split_dense_expected:
    @R.function
    def main(
        A: R.Tensor((16, 32), dtype="float16"), B: R.Tensor((32, 64), dtype="float16")
    ) -> R.Tensor(None, dtype="float16", ndim=2):
        # block 0
        gv = R.call_tir(HGEMM, (A, B), (16, 64), dtype="float16")
        return gv

    @T.prim_func
    def HGEMM(
        A: T.Buffer[(16, 32), "float16"],
        B: T.Buffer[(32, 64), "float16"],
        C: T.Buffer[(16, 64), "float16"],
    ):
        # function attr dict
        T.func_attr(
            {
                "cutlass_codegen": True,
                "global_symbol": "HGEMM",
                "cutlass_kernel": ["dense_row_row_row"],
            }
        )
        # body
        # with T.block("root")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads(A[v_3, v_5], B[v_5, v_4])
                T.writes(C[v_3, v_4])
                with T.init():
                    C[v_3, v_4] = T.float16(0)
                C[v_3, v_4] = C[v_3, v_4] + A[v_3, v_5] * B[v_5, v_4]


@tvm.script.ir_module
class test_cutlass_split_dense_relu_expected:
    @T.prim_func
    def unfused_epilogue(D: T.Buffer[(16, 64), "float16"], C: T.Buffer[(16, 64), "float16"]):
        # function attr dict
        T.func_attr({"global_symbol": "HGEMM"})
        # body
        # with T.block("root")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads()
                T.writes()
                with T.init():
                    D[v_3, v_4] = T.float16(0)
                T.evaluate(0)
        for v_6, v_7 in T.grid(16, 64):
            with T.block("root"):
                v_8, v_9 = T.axis.remap("SS", [v_6, v_7])
                T.reads(D[v_8, v_9])
                T.writes(C[v_8, v_9])
                C[v_8, v_9] = T.max(D[v_8, v_9], T.float16(0))

    @R.function
    def main(
        A: R.Tensor((16, 32), dtype="float16"), B: R.Tensor((32, 64), dtype="float16")
    ) -> R.Tensor(None, dtype="float16", ndim=2):
        # block 0
        gv = R.call_tir(cutlass_primfunc, (A, B), (16, 64), dtype="float16")
        gv1 = R.call_tir(unfused_epilogue, (gv,), (16, 64), dtype="float16")
        return gv1

    @T.prim_func
    def cutlass_primfunc(
        A: T.Buffer[(16, 32), "float16"],
        B: T.Buffer[(32, 64), "float16"],
        D: T.Buffer[(16, 64), "float16"],
    ):
        # function attr dict
        T.func_attr(
            {
                "cutlass_codegen": True,
                "global_symbol": "HGEMM",
                "cutlass_kernel": ["dense_row_row_row"],
            }
        )
        # buffer definition
        C = T.buffer_decl([16, 64], dtype="float16")
        # body
        # with T.block("root")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads(A[v_3, v_5], B[v_5, v_4])
                T.writes(D[v_3, v_4])
                with T.init():
                    D[v_3, v_4] = T.float16(0)
                D[v_3, v_4] = D[v_3, v_4] + A[v_3, v_5] * B[v_5, v_4]
        for v_6, v_7 in T.grid(16, 64):
            with T.block("root"):
                v_8, v_9 = T.axis.remap("SS", [v_6, v_7])
                T.reads(D[v_8, v_9])
                T.writes(C[v_8, v_9])
                C[v_8, v_9] = T.max(D[v_8, v_9], T.float16(0))


@tvm.script.ir_module
class test_cutlass_split_dense_bias_relu_expected:
    @T.prim_func
    def cutlass_primfunc(
        A: T.Buffer[(16, 32), "float16"],
        B: T.Buffer[(32, 64), "float16"],
        Bias: T.Buffer[(1, 16), "float16"],
        buf__2: T.Buffer[(16, 64), "float16"],
    ):
        # function attr dict
        T.func_attr(
            {"cutlass_codegen": True, "cutlass_kernel": ["dense_row_row_row", "bias_row", "relu"]}
        )
        # body
        # with T.block("root")
        buf_ = T.alloc_buffer([16, 64], dtype="float16")
        buf__1 = T.alloc_buffer([16, 64], dtype="float16")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads(A[v_3, v_5], B[v_5, v_4])
                T.writes(buf_[v_3, v_4])
                with T.init():
                    buf_[v_3, v_4] = T.float16(0)
                buf_[v_3, v_4] = buf_[v_3, v_4] + A[v_3, v_5] * B[v_5, v_4]
        for v_6, v_7 in T.grid(16, 64):
            with T.block("bias"):
                v_8, v_9 = T.axis.remap("SS", [v_6, v_7])
                T.reads(buf_[v_8, v_9], Bias[0, v_9])
                T.writes(buf__1[v_8, v_9])
                buf__1[v_8, v_9] = buf_[v_8, v_9] + Bias[0, v_9]
        for v_10, v_11 in T.grid(16, 64):
            with T.block("relu"):
                v_12, v_13 = T.axis.remap("SS", [v_10, v_11])
                T.reads(buf__1[v_12, v_13])
                T.writes(buf__2[v_12, v_13])
                buf__2[v_12, v_13] = T.max(buf__1[v_12, v_13], T.float16(0))
        for v_14, v_15 in T.grid(16, 64):
            with T.block("exp"):
                v_16, v_17 = T.axis.remap("SS", [v_14, v_15])
                T.reads()
                T.writes()
                T.evaluate(0)

    @R.function
    def main(
        A: R.Tensor((16, 32), dtype="float16"),
        B: R.Tensor((32, 64), dtype="float16"),
        Bias: R.Tensor((1, 16), dtype="float16"),
    ) -> R.Tensor(None, dtype="float16", ndim=2):
        # block 0
        gv = R.call_tir(cutlass_primfunc, (A, B, Bias), (16, 64), dtype="float16")
        gv1 = R.call_tir(unfused_epilogue, (gv,), (16, 64), dtype="float16")
        return gv1

    @T.prim_func
    def unfused_epilogue(buf__2: T.Buffer[(16, 64), "float16"], C: T.Buffer[(16, 64), "float16"]):
        # buffer definition
        buf_ = T.buffer_decl([16, 64], dtype="float16")
        # body
        # with T.block("root")
        for v, v_1, v_2 in T.grid(16, 64, 32):
            with T.block("dense_row_row_row"):
                v_3, v_4, v_5 = T.axis.remap("SSR", [v, v_1, v_2])
                T.reads()
                T.writes()
                with T.init():
                    buf_[v_3, v_4] = T.float16(0)
                T.evaluate(0)
        for v_6, v_7 in T.grid(16, 64):
            with T.block("bias"):
                v_8, v_9 = T.axis.remap("SS", [v_6, v_7])
                T.reads()
                T.writes()
                T.evaluate(0)
        for v_10, v_11 in T.grid(16, 64):
            with T.block("relu"):
                v_12, v_13 = T.axis.remap("SS", [v_10, v_11])
                T.reads()
                T.writes()
                T.evaluate(0)
        for v_14, v_15 in T.grid(16, 64):
            with T.block("exp"):
                v_16, v_17 = T.axis.remap("SS", [v_14, v_15])
                T.reads(buf__2[v_16, v_17])
                T.writes(C[v_16, v_17])
                C[v_16, v_17] = T.exp(buf__2[v_16, v_17], dtype="float16")


def test_cutlass_split_dense():
    mod = GEMM
    mod = relax.transform.SplitCutlass()(mod)
    mod = relax.transform.RemoveUnusedFunctions()(mod)
    tvm.ir.assert_structural_equal(mod, test_cutlass_split_dense_expected)


def test_cutlass_split_dense_relu():
    mod = GEMM_ReLU
    mod = relax.transform.SplitCutlass()(mod)
    mod = relax.transform.RemoveUnusedFunctions()(mod)
    tvm.ir.assert_structural_equal(mod, test_cutlass_split_dense_relu_expected)


def test_cutlass_split_dense_bias_relu():
    mod = GEMM_bias_relu_exp
    mod = relax.transform.SplitCutlass()(mod)
    mod = relax.transform.RemoveUnusedFunctions()(mod)
    tvm.ir.assert_structural_equal(mod, test_cutlass_split_dense_bias_relu_expected)


if __name__ == "__main__":
    test_cutlass_split_dense()
    test_cutlass_split_dense_relu()
    test_cutlass_split_dense_bias_relu()
