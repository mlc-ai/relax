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
import pytest
from tvm.relay.expr import const
import tvm
import tvm.testing
from tvm import relax
import tvm.script
import numpy as np
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import tir as T
from tvm.tir import StringImm, cutlass_gemm



GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


def construct_mod(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "cutlass_codegen": 1,
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE)
                          )  # pylint: disable=invalid-name
                with T.block("cutlass"):
                    T.reads(A[0:m, 0:k], B[0:k, 0:n])
                    T.writes(C[0:m, 0:n])
                    T.evaluate(
                        cutlass_gemm(
                            A.data,
                            B.data,
                            C.data,
                            StringImm(A.dtype),
                            StringImm(B.dtype),
                            StringImm(C.dtype),
                            transpose_a=False,
                            transpose_b=False,
                            transpose_c=False,
                        )
                    )
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                C = R.call_tir(frame.global_vars[GLOBAL_SYMBOL], args=[
                               A, B], shape=(m, n), dtype=C_TYPE)
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def construct_mod_gemm_relu(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "cutlass_codegen": 1,
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE)
                          )  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.block("cutlass"):
                    T.reads(A[0:m, 0:k], B[0:k, 0:n])
                    T.writes(D[0:m, 0:n])
                    T.evaluate(
                        cutlass_gemm(
                            A.data,
                            B.data,
                            D.data,
                            StringImm(A.dtype),
                            StringImm(B.dtype),
                            StringImm(D.dtype),
                            transpose_a=False,
                            transpose_b=False,
                            transpose_c=False,
                        )
                    )
                with T.grid(16, 64) as (i, j):
                    with T.block("relu"):
                        T.reads(D[i, j])
                        T.writes(C[i, j])
                        T.buffer_store(C, T.max(D[i, j], T.cast(0, "float16")), [i, j])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                C = R.call_tir(frame.global_vars[GLOBAL_SYMBOL], args=[
                               A, B], shape=(m, n), dtype=C_TYPE)
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def construct_mod_gemm_relu(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "cutlass_codegen": 1,
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE)
                          )  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.block("cutlass"):
                    T.reads(A[0:m, 0:k], B[0:k, 0:n])
                    T.writes(D[0:m, 0:n])
                    T.evaluate(
                        cutlass_gemm(
                            A.data,
                            B.data,
                            D.data,
                            StringImm(A.dtype),
                            StringImm(B.dtype),
                            StringImm(D.dtype),
                            transpose_a=False,
                            transpose_b=False,
                            transpose_c=False,
                        )
                    )
                with T.grid(16, 64) as (i, j):
                    with T.block("relu"):
                        T.reads(D[i, j])
                        T.writes(C[i, j])
                        T.buffer_store(
                            C, T.max(D[i, j], T.cast(0, "float16")), [i, j])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                C = R.call_tir(frame.global_vars[GLOBAL_SYMBOL], args=[
                               A, B], shape=(m, n), dtype=C_TYPE)
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def construct_mod_gemm_bias_relu_exp(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "cutlass_codegen": 1,
                        # "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                Bias = T.arg("Bias", T.buffer_decl((m,), C_TYPE))
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE)
                          )  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                E = T.alloc_buffer((m, n), C_TYPE)
                F = T.alloc_buffer((m, n), C_TYPE)
                with T.block("cutlass"):
                    T.reads(A[0:m, 0:k], B[0:k, 0:n])
                    T.writes(D[0:m, 0:n])

                    T.evaluate(
                        cutlass_gemm(
                            A.data,
                            B.data,
                            D.data,
                            StringImm(A.dtype),
                            StringImm(B.dtype),
                            StringImm(D.dtype),
                            transpose_a=False,
                            transpose_b=False,
                            transpose_c=False,
                        )
                    )
                with T.grid(16, 64) as (i, j):
                    with T.block("bias"):
                        T.reads(D[i, j], Bias[i])
                        T.writes(E[i, j])
                        T.buffer_store(
                            E, D[i, j] + Bias[i], [i, j])
                with T.grid(16, 64) as (i, j):
                    with T.block("relu"):
                        T.reads(E[i, j])
                        T.writes(F[i, j])
                        T.buffer_store(
                            F, T.max(E[i, j], T.cast(0, "float16")), [i, j])
                with T.grid(16, 64) as (i, j):
                    with T.block("exp"):
                        T.reads(F[i, j])
                        T.writes(C[i, j])
                        T.buffer_store(
                            C, T.exp(F[i, j]), [i, j])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE)
                          )  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE)
                          )  # pylint: disable=invalid-name
                Bias = R.arg("Bias", R.tensor((m,), C_TYPE))
                C = R.call_tir(frame.global_vars[GLOBAL_SYMBOL], args=[
                               A, B, Bias], shape=(m, n), dtype=C_TYPE)
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_split_dense():
    m, n, k = 16, 64, 32
    mod = construct_mod(m=m, n=n, k=k)
    print(mod.script())
    new_mod = relax.transform.SplitCutlass()(mod)
    new_mod = relax.transform.RemoveUnusedFunctions()(new_mod)
    print(new_mod.script())


def test_cutlass_split_dense_bias_relu():
    m, n, k = 16, 64, 32
    mod = construct_mod_gemm_bias_relu_exp(m=m, n=n, k=k)
    print(mod.script())
    new_mod = relax.transform.SplitCutlass()(mod)
    new_mod = relax.transform.RemoveUnusedFunctions()(new_mod)
    print(new_mod.script())


def test_cutlass_split_fail_dense_relu():
    m, n, k = 16, 64, 32
    mod = construct_mod_gemm_relu(m=m, n=n, k=k)
    new_mod = relax.transform.SplitCutlass()(mod)
    print(new_mod.script())


if __name__ == "__main__":
    # test_cutlass_split_dense()
    # test_cutlass_split_fail_dense_relu()
    test_cutlass_split_dense_bias_relu()
