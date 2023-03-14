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

import tvm
from tvm.script import tir as T, relax as R
from tvm import relax
import tvm.testing
import numpy as np


# fmt: off
@tvm.script.ir_module
class Module:
    @T.prim_func
    def reshape(rxplaceholder: T.Buffer[(T.int64(2), T.int64(4)), "float32"], T_reshape: T.Buffer[T.int64(8), "float32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "reshape"})
        # body
        # with T.block("root")
        for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                    T.reads(rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)])
                    T.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[T.Cast("int64", ax0) % T.int64(8) // T.int64(4), T.Cast("int64", ax0) % T.int64(4)]

    @T.prim_func
    def exp(rxplaceholder: T.Buffer[(T.int64(2), T.int64(4)), "float32"], compute: T.Buffer[(T.int64(2), T.int64(4)), "float32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "exp"})
        # body
        # with T.block("root")
        for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                with T.block("compute"):
                    i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                    i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                    T.reads(rxplaceholder[i0, i1])
                    T.writes(compute[i0, i1])
                    compute[i0, i1] = T.exp(rxplaceholder[i0, i1], dtype="float32")

    @T.prim_func
    def add(rxplaceholder: T.Buffer[T.int64(8), "float32"], rxplaceholder_1: T.Buffer[(), "float32"], T_add: T.Buffer[T.int64(8), "float32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "add"})
        # body
        # with T.block("root")
        for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                with T.block("T_add"):
                    ax0 = T.axis.spatial(T.int64(8), i0_fused_0 * T.int64(8) + i0_fused_1)
                    T.reads(rxplaceholder[ax0], rxplaceholder_1[()])
                    T.writes(T_add[ax0])
                    T_add[ax0] = rxplaceholder[ax0] + rxplaceholder_1[()]

    @T.prim_func
    def pad(rxplaceholder: T.Buffer[T.int64(8), "float32"], PadInput: T.Buffer[T.int64(10), "float32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "pad"})
        # body
        # with T.block("root")
        for i0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(T.int64(10), thread="threadIdx.x"):
                with T.block("PadInput"):
                    i0 = T.axis.spatial(T.int64(10), i0_fused_0 * T.int64(10) + i0_fused_1)
                    T.reads(rxplaceholder[i0 - T.int64(1)])
                    T.writes(PadInput[i0])
                    PadInput[i0] = T.if_then_else(T.int64(1) <= i0 and i0 < T.int64(9), rxplaceholder[i0 - T.int64(1)], T.float32(1), dtype="float32")

    @R.function
    def cuda_graph_capture_func_alloc() -> R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")):
        # function attr dict
        R.func_attr({"global_symbol": "cuda_graph_capture_func_alloc"})
        # block 0
        shape_heap1: R.Object = R.null_value()
        storage: R.Object = R.vm.alloc_storage((32,), dtype="float32", runtime_device_index=0)
        gv3: R.Tensor((2, 4), dtype="float32") = R.vm.alloc_tensor(storage, (2, 4), offset=0, dtype="float32")
        storage1: R.Object = R.vm.alloc_storage((32,), dtype="float32", runtime_device_index=0)
        gv11: R.Tensor((8,), dtype="float32") = R.vm.alloc_tensor(storage1, (8,), offset=0, dtype="float32")
        storage2: R.Object = R.vm.alloc_storage((32,), dtype="float32", runtime_device_index=0)
        gv21: R.Tensor((8,), dtype="float32") = R.vm.alloc_tensor(storage2, (8,), offset=0, dtype="float32")
        gv31: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")) = (gv3, gv11, gv21)
        return gv31

    @R.function
    def cuda_graph_capture_func_capture(allocs: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))) -> R.Tuple():
        # function attr dict
        R.func_attr({"global_symbol": "cuda_graph_capture_func_capture"})
        # block 0
        shape_heap: R.Object = R.null_value()
        _: R.Tuple() = R.call_packed("vm.builtin.check_tuple_info", allocs, 3, "", sinfo_args=[R.Tuple()])
        gv: R.Tensor((2, 4), dtype="float32") = allocs[0]
        _1: R.Tuple() = R.call_packed("vm.builtin.check_tensor_info", gv, 2, "", sinfo_args=[R.Tuple()])
        gv1: R.Tensor((8,), dtype="float32") = allocs[1]
        _2: R.Tuple() = R.call_packed("vm.builtin.check_tensor_info", gv1, 1, "", sinfo_args=[R.Tuple()])
        gv2: R.Tensor((8,), dtype="float32") = allocs[2]
        _3: R.Tuple() = R.call_packed("vm.builtin.check_tensor_info", gv2, 1, "", sinfo_args=[R.Tuple()])
        _4: R.Tuple() = R.call_packed("vm.builtin.match_shape", gv, shape_heap, 2, 0, 2, 0, 4, "", sinfo_args=[R.Tuple()])
        _5: R.Tuple() = R.call_packed("vm.builtin.match_shape", gv1, shape_heap, 1, 0, 8, "", sinfo_args=[R.Tuple()])
        _6: R.Tuple() = R.call_packed("vm.builtin.match_shape", gv2, shape_heap, 1, 0, 8, "", sinfo_args=[R.Tuple()])
        alloc: R.Tensor((2, 4), dtype="float32") = allocs[0]
        alloc1: R.Tensor((8,), dtype="float32") = allocs[1]
        alloc2: R.Tensor((8,), dtype="float32") = allocs[2]
        _11: R.Tuple() = reshape(alloc, alloc1)
        gv0: R.Tensor((), dtype="float32") = R.const(1, "float32")
        _21: R.Tuple() = add(alloc1, gv0, alloc2)
        gv4: R.Tuple() = R.tuple()
        return gv4

    @R.function
    def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        # function attr dict
        R.func_attr({"global_symbol": "main"})
        # block 0
        shape_heap2: R.Object = R.null_value()
        _7: R.Tuple() = R.call_packed("vm.builtin.check_tensor_info", x, 2, "float32", "", sinfo_args=[R.Tuple()])
        _8: R.Tuple() = R.call_packed("vm.builtin.match_shape", x, shape_heap2, 2, 0, 2, 0, 4, "", sinfo_args=[R.Tuple()])
        gv5: R.Tuple(R.Object, R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))) = R.call_builtin_with_ctx("vm.builtin.get_captured_cuda_graph", (cuda_graph_capture_func_alloc, cuda_graph_capture_func_capture), sinfo_args=[R.Tuple(R.Object, R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")))])
        gv12: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")) = gv5[1]
        gv22: R.Object = gv5[0]
        gv32: R.Tensor((2, 4), dtype="float32") = gv12[0]
        _9: R.Tuple() = exp(x, gv32)
        gv41: R.Tuple() = R.call_packed("vm.builtin.cuda_graph_launch", gv22, sinfo_args=[R.Tuple()])
        storage3: R.Object = R.vm.alloc_storage((40,), dtype="float32", runtime_device_index=0)
        alloc3: R.Tensor((10,), dtype="float32") = R.vm.alloc_tensor(storage3, (10,), offset=0, dtype="float32")
        gv51: R.Tensor((8,), dtype="float32") = gv12[2]
        _31: R.Tuple() = pad(gv51, alloc3)
        return alloc3

# fmt: on


def codegen(mod, target, exec_mode="bytecode"):
    builder = relax.ExecBuilder()
    tir_mod = relax.vm._vmcodegen(builder, mod, exec_mode=exec_mode)
    return relax.vm._vmlink(builder, target, tir_mod)


@tvm.testing.requires_cuda
def test_vm_run():
    mod = Module
    target = tvm.target.Target("cuda", host="llvm")
    ex = codegen(mod, target)
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(ex, dev)
    x_np = np.random.uniform(size=(2, 4)).astype("float32")
    y_np = np.exp(x_np)
    y_np = y_np.reshape((8,))
    y_np = y_np + 1.0
    pad_value = np.ones((1,)).astype("float32")
    y_np = np.concatenate([pad_value, y_np, pad_value], axis=0)

    x = tvm.nd.array(x_np, dev)
    y = vm["main"](x)
    tvm.testing.assert_allclose(y.asnumpy(), y_np, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
