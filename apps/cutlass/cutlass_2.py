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
                        "global_symbol": GLOBAL_SYMBOL,
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


def main():
    m, n, k = 16, 64, 32
    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    mod = construct_mod(m=m, n=n, k=k)
    with tvm.transform.PassContext():
        mod = relax.transform.CutlassCodegen()(mod)
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
    # print(c)
    # print(c_np)
    np.testing.assert_allclose(c.numpy(), c_np, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
