# pylint: disable=missing-docstring
from __future__ import annotations

import numpy as np
import tvm
from tvm import relax
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import tir as T
from tvm.tir import StringImm, cutlass_gemm

PKG_FILE = "/tmp/packaged.so"
GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


def construct_mod(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module():
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "cutlass_codegen": 1,
                        "cutlass_kernel": "cutlass.dense"
                        # "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
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
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(GLOBAL_SYMBOL, args=[A, B], shape=(m, n), dtype=C_TYPE)
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def construct_mod_gemm_bias_relu(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "cutlass_codegen": 1,
                        "cutlass_kernel": "cutlass.dense_bias_relu",
                        # "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                Bias = T.arg("Bias", T.buffer_decl((1, n), C_TYPE))
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name

                D = T.alloc_buffer((m, n), C_TYPE)
                E = T.alloc_buffer((m, n), C_TYPE)
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
                with T.grid(m, n) as (i, j):
                    with T.block("bias"):
                        T.reads(D[i, j], Bias[0, j])
                        T.writes(E[i, j])
                        T.buffer_store(E, D[i, j] + Bias[0, j], [i, j])
                with T.grid(m, n) as (i, j):
                    with T.block("relu"):
                        T.reads(E[i, j])
                        T.writes(C[i, j])
                        T.buffer_store(C, T.max(E[i, j], T.cast(0, "float16")), [i, j])
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((k, n), B_TYPE))  # pylint: disable=invalid-name
                Bias = R.arg("Bias", R.tensor((1, n), C_TYPE))
                C = R.call_tir(
                    frame.global_vars[GLOBAL_SYMBOL], args=[A, B, Bias], shape=(m, n), dtype=C_TYPE
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def gemm():
    m, n, k = 16, 64, 32
    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    mod = construct_mod(m=m, n=n, k=k)

    print(mod.script())

    with tvm.transform.PassContext():
        mod = relax.transform.CutlassCodegen()(mod)
        print(mod.script())
        # print("attrs['c_source']:", mod[GLOBAL_SYMBOL].attrs["c_source"])
        # print("attrs['c_source_fmt']:", mod[GLOBAL_SYMBOL].attrs["c_source_fmt"])
        exe = relax.vm.build(mod, target=target)
    exe.mod.export_library(PKG_FILE, cc="nvcc")
    vm = relax.VirtualMachine(
        tvm.runtime.load_module(PKG_FILE),
        tvm.cuda(),
    )
    a_np = np.random.rand(m, k).astype(A_TYPE)
    b_np = np.random.rand(k, n).astype(B_TYPE)
    c_np = np.matmul(a_np, b_np)
    a = tvm.nd.array(a_np, device=tvm.cuda())
    b = tvm.nd.array(b_np, device=tvm.cuda())
    c = vm["main"](a, b)
    tvm.cuda().sync()
    print(c)
    print(c_np)
    np.testing.assert_allclose(c.numpy(), c_np, rtol=1e-2, atol=1e-2)


def gemm_bias_relu():
    m, n, k = 16, 64, 32
    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    mod = construct_mod_gemm_bias_relu(m=m, n=n, k=k)

    print(mod.script())

    with tvm.transform.PassContext():
        mod = relax.transform.CutlassCodegen()(mod)
        print(mod.script())
        # print("attrs['c_source']:", mod[GLOBAL_SYMBOL].attrs["c_source"])
        # print("attrs['c_source_fmt']:", mod[GLOBAL_SYMBOL].attrs["c_source_fmt"])
        executable = relax.vm.build(mod, target=target)
    executable.mod.export_library(PKG_FILE, cc="nvcc")
    executable = tvm.runtime.load_module(PKG_FILE)
    vm = relax.VirtualMachine(executable, tvm.cuda())
    a_np = np.random.rand(16, 32).astype("float16")
    b_np = np.random.rand(32, 64).astype("float16")
    bias_np = np.random.rand(1, 64).astype("float16")

    a = tvm.nd.array(a_np, device=tvm.cuda())
    b = tvm.nd.array(b_np, device=tvm.cuda())
    bias = tvm.nd.array(bias_np, device=tvm.cuda())
    c = vm["main"](a, b, bias)
    c_np = np.maximum(np.matmul(a_np, b_np) + bias_np, 0)
    print(c)
    print(c_np)

    np.testing.assert_allclose(c.numpy(), c.numpy(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    gemm()
    gemm_bias_relu()
