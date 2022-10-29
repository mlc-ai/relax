from __future__ import annotations

import numpy as np
import tvm
from tvm import relax, te
from tvm.contrib import cublas
from tvm.script import relax as R
from tvm.script import tir as T

SRC_FILE = "./apps/cutlass/gemm.cu"
SRC_FILE2 = "./apps/cutlass/gemm_bias_relu.cu"
PKG_FILE = "/tmp/packaged.so"


@tvm.script.ir_module
class TestModule:
    # Input IRModule.
    @R.function
    def main(
        A: Tensor((16, 32), "float16"),
        B: Tensor((32, 64), "float16"),
    ):
        C = relax.call_tir(
            "HGEMM",
            (A, B),
            (16, 64),
            dtype="float16",
        )
        return C


@tvm.script.ir_module
class TestModule2:
    # Input IRModule.
    @R.function
    def main(
        A: Tensor((16, 32), "float16"),
        B: Tensor((32, 64), "float16"),
        Bias: Tensor((1, 64), "float16"),
    ):
        C = relax.call_tir(
            "HGEMM",
            (A, B, Bias),
            (16, 64),
            dtype="float16",
        )
        return C


def import_source_module(executable, src_file=SRC_FILE):
    code = open(src_file, "r").read()
    fmt = "cu"
    func_names = ["whatever.cu"]
    const_vars = []  # type: ignore
    mod = tvm.get_global_func("runtime.CSourceModuleCreate")(
        code,
        fmt,
        func_names,
        const_vars,
    )
    executable.mod.import_module(mod)


def gemm():
    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    with target:
        executable = relax.vm.build(TestModule, target=target)
        import_source_module(executable)
        executable.mod.export_library(
            PKG_FILE,
            cc="nvcc",
        )
    executable = tvm.runtime.load_module(PKG_FILE)
    vm = relax.VirtualMachine(executable, tvm.cuda())
    a_np = np.random.rand(16, 32).astype("float16")
    b_np = np.random.rand(32, 64).astype("float16")

    a = tvm.nd.array(a_np, device=tvm.cuda())
    b = tvm.nd.array(b_np, device=tvm.cuda())
    c = vm["main"](a, b)
    print(c)
    
    A = te.placeholder((16, 32), name="A", dtype="float16")
    B = te.placeholder((32, 64), name="B", dtype="float16")
    C = cublas.matmul(A, B, dtype="float16")
    s = te.create_schedule(C.op)
    dev = tvm.cuda(0)
    f = tvm.build(s, [A, B, C], target)
    a_cublas = tvm.nd.array(a_np, dev)
    b_cublas = tvm.nd.array(b_np, dev)
    c_cublas = tvm.nd.array(np.zeros((16, 64), dtype=C.dtype), dev)
    f(a_cublas, b_cublas, c_cublas)
    print(c_cublas)

    np.testing.assert_allclose(c.numpy(), c_cublas.numpy(), rtol=1e-2, atol=1e-2)


def gemm_bias_relu():
    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    with target:
        executable = relax.vm.build(TestModule2, target=target)
        import_source_module(executable, SRC_FILE2)
        executable.mod.export_library(
            PKG_FILE,
            cc="nvcc",
        )
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
