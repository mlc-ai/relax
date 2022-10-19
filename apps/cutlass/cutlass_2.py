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
M = 256
N = 512
K = 1024
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"
TARGET = tvm.target.Target("nvidia/geforce-rtx-3090-ti")


def cublass_matmul(a_np, b_np):
    from tvm import te
    from tvm.contrib import cublas

    A = te.placeholder((M, K), name="A", dtype=A_TYPE)
    B = te.placeholder((K, N), name="B", dtype=B_TYPE)
    C = cublas.matmul(A, B, transa=False, transb=False, dtype=C_TYPE)
    s = te.create_schedule(C.op)

    if not tvm.get_global_func("tvm.contrib.cublas.matmul", True):
        raise ValueError("skip because extern function is not available")
    dev = tvm.cuda(0)
    f = tvm.build(s, [A, B, C], target=TARGET)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype=C.dtype), dev)
    f(a, b, c)
    return c.numpy()


def construct_mod():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module():
            with T.prim_func():
                T.func_name(GLOBAL_SYMBOL)
                T.func_attr(
                    {
                        "cutlass_codegen": 1,
                        "global_symbol": GLOBAL_SYMBOL,
                    }
                )
                A = T.arg("A", T.buffer_decl((M, K), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((K, N), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((M, N), C_TYPE))  # pylint: disable=invalid-name
                with T.block("cutlass"):
                    T.reads(A[0:M, 0:K], B[0:K, 0:N])
                    T.writes(C[0:M, 0:N])
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
                A = R.arg("A", R.tensor((M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                C = R.call_tir(GLOBAL_SYMBOL, args=[A, B], shape=(M, N), dtype=C_TYPE)
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def main():
    mod = construct_mod()
    with tvm.transform.PassContext():
        mod = relax.transform.CutlassCodegen()(mod)
        # print("attrs['c_source']:", mod[GLOBAL_SYMBOL].attrs["c_source"])
        # print("attrs['c_source_fmt']:", mod[GLOBAL_SYMBOL].attrs["c_source_fmt"])
        exe = relax.vm.build(mod, target=TARGET)
    exe.mod.export_library(PKG_FILE, cc="nvcc")
    vm = relax.VirtualMachine(
        tvm.runtime.load_module(PKG_FILE),
        tvm.cuda(),
    )
    a_np = np.random.rand(M, K).astype(A_TYPE)
    b_np = np.random.rand(K, N).astype(B_TYPE)
    c_np = cublass_matmul(a_np, b_np)
    a = tvm.nd.array(a_np, device=tvm.cuda())
    b = tvm.nd.array(b_np, device=tvm.cuda())
    c = vm["main"](a, b)
    tvm.cuda().sync()
    # print(c)
    # print(c_np)
    np.testing.assert_allclose(c.numpy(), c_np, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
