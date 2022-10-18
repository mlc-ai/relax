from __future__ import annotations

import numpy as np
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script import tir as T

SRC_FILE = "./apps/cutlass/cutlass.cu"
PKG_FILE = "/tmp/packaged.so"


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


def import_source_module(executable):
    code = open(SRC_FILE, "r").read()
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


def main():
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
    a = tvm.nd.array(np.random.rand(16, 32).astype("float16"), device=tvm.cuda())
    b = tvm.nd.array(np.random.rand(32, 64).astype("float16"), device=tvm.cuda())
    c = vm["main"](a, b)
    print(c)


if __name__ == "__main__":
    main()
