# pylint: disable=missing-docstring
from __future__ import annotations

import numpy as np
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as T
from tvm.tir import StringImm, cutlass_gemm

PKG_FILE = "/tmp/packaged.so"
GLOBAL_SYMBOL = "HGEMM"


@tvm.script.ir_module
class TestModule:
    # Input IRModule.
    @R.function
    def main(
        A: Tensor((16, 32), "float32"),
        B: Tensor((32, 64), "float32"),
    ):
        C = relax.call_tir(
            "HGEMM",
            (A, B),
            (16, 64),
            dtype="float16",
        )
        return C


def construct_mod(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with T.prim_func():
            T.func_attr(
                {
                    "cutlass_codegen": 1,
                    "global_symbol": GLOBAL_SYMBOL,
                }
            )
            A = T.arg("A", T.buffer_decl((m, k), "float32"))  # pylint: disable=invalid-name
            B = T.arg("B", T.buffer_decl((k, n), "float32"))  # pylint: disable=invalid-name
            C = T.arg("C", T.buffer_decl((m, n), "float32"))  # pylint: disable=invalid-name
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
    mod = tvm.IRModule(
        {
            GLOBAL_SYMBOL: ib.get(),
            "main": TestModule["main"],
        }
    )
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
    a = tvm.nd.array(np.random.rand(m, k).astype("float16"), device=tvm.cuda())
    b = tvm.nd.array(np.random.rand(k, n).astype("float16"), device=tvm.cuda())
    c = vm["main"](a, b)


main()
