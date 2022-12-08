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
import tempfile

from tvm import relax, runtime
import tvm
import tvm.testing
from tvm import relax
import numpy as np
from tvm.relax.vm import build as relax_build

import tvm.relax.cutlass.pattern

PKG_FILE = "/tmp/test_transform_cutlass_codegen.so"
GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"

target = "cuda"


def f_run(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    return vm["main"](*input)


def build(mod):
    mod = relax.transform.SplitCutlass()(mod)
    print(mod.script())
    mod = relax.transform.CutlassCodegen()(mod)
    print(mod.script())
    executbale = relax_build(mod, target)
    executbale.mod.export_library(PKG_FILE, cc="nvcc")
    return executbale


def constructGEMM(m, n, k, GLOBAL_SYMBOL="HGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(D, D[vi, vj] + A[vi, vk] * B[vk, vj], [vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B], shape=(m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_dense():
    m, n, k = 128, 128, 128
    build(constructGEMM(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructGEMM_bias(m, n, k, GLOBAL_SYMBOL="HGEMM_bias"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = T.arg("bias", T.buffer_decl((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(D, D[vi, vj] + A[vi, vk] * B[vk, vj], [vi, vj])
                with T.grid(m, n) as (i, j):
                    with T.block("bias"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.reads(D[vi, vj], bias[0, vj])
                        T.writes(C[vi, vj])
                        T.buffer_store(C, D[vi, vj] + bias[0, vj], [vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B, bias], shape=(m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_dense_bias():
    m, n, k = 128, 128, 128
    build(constructGEMM_bias(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructGEMM_bias_relu(m, n, k, GLOBAL_SYMBOL="HGEMM"):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = T.arg("bias", T.buffer_decl((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
                D = T.alloc_buffer((m, n), C_TYPE)
                E = T.alloc_buffer((m, n), C_TYPE)
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(D[vi, vj])
                        with T.init():
                            T.buffer_store(D, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(D, D[vi, vj] + A[vi, vk] * B[vk, vj], [vi, vj])
                with T.grid(m, n) as (i, j):
                    with T.block("bias"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.reads(D[vi, vj], bias[0, vj])
                        T.writes(E[vi, vj])
                        T.buffer_store(E, D[vi, vj] + bias[0, vj], [vi, vj])
                with T.grid(m, n) as (i, j):
                    with T.block("relu"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        T.reads(E[vi, vj])
                        T.writes(C[vi, vj])
                        T.buffer_store(C, T.max(E[vi, vj], T.cast(0.0, C_TYPE)), [vi, vj])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, n), A_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B, bias], shape=(m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_dense_bias_relu():
    m, n, k = 128, 128, 128
    build(constructGEMM_bias_relu(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), np.maximum(A @ B + bias, 0), rtol=1e-2)


if __name__ == "__main__":
    test_cutlass_dense()
    test_cutlass_dense_bias()
    test_cutlass_dense_bias_relu()
