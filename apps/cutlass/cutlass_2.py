# pylint: disable=missing-docstring
import tvm
from tvm import relax
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as T
from tvm.tir import StringImm, cutlass_gemm


def construct_mod(m, n, k):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with T.prim_func():
            T.func_attr(
                {
                    "cutlass_codegen": 1,
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
    mod = tvm.IRModule({"main": ib.get()})
    return mod


def main():
    mod = construct_mod(m=16, n=64, k=32)
    with tvm.transform.PassContext():
        mod = relax.transform.CutlassCodegen()(mod)
    print("attrs['c_source']:", mod["main"].attrs["c_source"])
    print("attrs['cutlass_codegen']:", mod["main"].attrs["cutlass_codegen"])
    mod.show()


main()
