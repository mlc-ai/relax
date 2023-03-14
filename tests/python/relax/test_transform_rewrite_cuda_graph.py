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
from tvm import relax
from tvm.script import tir as T, relax as R, ir as I
import tvm.testing


def test_rewrite_cuda_graph():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer(T.int64(8), "float32")):
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
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
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
        def add(rxplaceholder: T.Buffer(T.int64(8), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer(T.int64(8), "float32")):
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
        def pad(rxplaceholder: T.Buffer(T.int64(8), "float32"), PadInput: T.Buffer(T.int64(10), "float32")):
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
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            cls = Before
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([8]), "float32")
            _1: R.Tuple = cls.reshape(alloc, alloc1)
            __1: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([8]), "float32")
            gv0: R.Tensor((), dtype="float32") = R.const(1, "float32")
            _2: R.Tuple = cls.add(alloc1, gv0, alloc2)
            _1_1: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), "float32", 0)
            _3: R.Tuple = cls.pad(alloc2, alloc3)
            _2_1: R.Tuple = R.memory.kill_tensor(alloc2)
            _3_1: R.Tuple = R.memory.kill_storage(storage)
            _4: R.Tuple = R.memory.kill_storage(storage1)
            return alloc3


    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer(T.int64(8), "float32")):
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
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
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
        def add(rxplaceholder: T.Buffer(T.int64(8), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer(T.int64(8), "float32")):
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
        def pad(rxplaceholder: T.Buffer(T.int64(8), "float32"), PadInput: T.Buffer(T.int64(10), "float32")):
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
        def cuda_graph_capture_func_alloc() -> R.Tuple(R.Object, R.Object):
            gv: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv1: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv2: R.Tuple(R.Object, R.Object) = (gv, gv1)
            return gv2

        @R.function
        def cuda_graph_capture_func_capture(allocs: R.Tuple(R.Object, R.Object)) -> R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")):
            cls = Expected
            storage: R.Object = allocs[0]
            storage1: R.Object = allocs[1]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([8]), R.dtype("float32"))
            _1: R.Tuple = cls.reshape(alloc, alloc1)
            alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([8]), R.dtype("float32"))
            gv0: R.Tensor((), dtype="float32") = R.const(1, "float32")
            _2: R.Tuple = cls.add(alloc1, gv0, alloc2)
            gv3: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")) = (alloc, alloc1, alloc2)
            return gv3

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            cls = Expected
            gv: R.Tuple(R.Object, R.Tuple(R.Object, R.Object), R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))) = R.call_builtin_with_ctx("vm.builtin.get_captured_cuda_graph", (cls.cuda_graph_capture_func_alloc, cls.cuda_graph_capture_func_capture), sinfo_args=(R.Tuple(R.Object, R.Tuple(R.Object, R.Object), R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32"))),))
            gv1: R.Tuple(R.Object, R.Object) = gv[1]
            gv2: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((8,), dtype="float32"), R.Tensor((8,), dtype="float32")) = gv[2]
            gv3: R.Object = gv[0]
            gv4: R.Tensor((2, 4), dtype="float32") = gv2[0]
            _: R.Tuple = cls.exp(x, gv4)
            gv5: R.Tuple = R.call_packed("vm.builtin.cuda_graph_launch", gv3, sinfo_args=(R.Tuple,))
            __1: R.Tuple = R.memory.kill_tensor(gv4)
            gv6: R.Tensor((8,), dtype="float32") = gv2[1]
            _1_1: R.Tuple = R.memory.kill_tensor(gv6)
            alloc3: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0))
            gv7: R.Tensor((8,), dtype="float32") = gv2[2]
            _3: R.Tuple = cls.pad(gv7, alloc3)
            _2_1: R.Tuple = R.memory.kill_tensor(gv7)
            gv8: R.Object = gv1[0]
            _3_1: R.Tuple = R.memory.kill_storage(gv8)
            gv9: R.Object = gv1[1]
            _4: R.Tuple = R.memory.kill_storage(gv9)
            return alloc3

    # fmt: on
    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_tuple():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
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


        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            cls = Before
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([2, 4]), "float32")
            lv0 = (alloc1,)
            lv1 = (lv0,)
            lv2 = lv1[0]
            lv3 = lv2[0]
            _1: R.Tuple = cls.exp(alloc, lv3)
            _2: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _3: R.Tuple = cls.exp(alloc1, alloc2)
            _4: R.Tuple = R.memory.kill_tensor(alloc1)
            _5: R.Tuple = R.memory.kill_storage(storage)
            _6: R.Tuple = R.memory.kill_storage(storage1)
            return alloc2

    @I.ir_module
    class Expected:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"global_symbol": "exp", "tir.noalias": True})
            # with T.block("root"):
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        T.reads(rxplaceholder[i0, i1])
                        T.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function
        def cuda_graph_capture_func_alloc() -> R.Tuple(R.Object, R.Object):
            gv: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv1: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv2: R.Tuple(R.Object, R.Object) = (gv, gv1)
            return gv2

        @R.function
        def cuda_graph_capture_func_capture(allocs: R.Tuple(R.Object, R.Object)) -> R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32"), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tensor((2, 4), dtype="float32")):
            cls = Expected
            storage: R.Object = allocs[0]
            storage1: R.Object = allocs[1]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            lv0: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (alloc1,)
            lv1: R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))) = (lv0,)
            lv2: R.Tuple(R.Tensor((2, 4), dtype="float32")) = lv1[0]
            lv3: R.Tensor((2, 4), dtype="float32") = lv2[0]
            _1: R.Tuple = cls.exp(alloc, lv3)
            gv3: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32"), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tensor((2, 4), dtype="float32")) = (alloc, alloc1, lv0, lv1, lv2, lv3)
            return gv3

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            cls = Expected
            gv: R.Tuple(R.Object, R.Tuple(R.Object, R.Object), R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32"), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tensor((2, 4), dtype="float32"))) = R.call_builtin_with_ctx("vm.builtin.get_captured_cuda_graph", (cls.cuda_graph_capture_func_alloc, cls.cuda_graph_capture_func_capture), sinfo_args=(R.Tuple(R.Object, R.Tuple(R.Object, R.Object), R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32"), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tensor((2, 4), dtype="float32"))),))
            gv1: R.Tuple(R.Object, R.Object) = gv[1]
            gv2: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32"), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))), R.Tuple(R.Tensor((2, 4), dtype="float32")), R.Tensor((2, 4), dtype="float32")) = gv[2]
            gv3: R.Object = gv[0]
            gv4: R.Tensor((2, 4), dtype="float32") = gv2[0]
            _: R.Tuple = cls.exp(x, gv4)
            gv5: R.Tuple = R.call_packed("vm.builtin.cuda_graph_launch", gv3, sinfo_args=(R.Tuple,))
            _2: R.Tuple = R.memory.kill_tensor(gv4)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            gv6: R.Tensor((2, 4), dtype="float32") = gv2[1]
            _3: R.Tuple = cls.exp(gv6, alloc2)
            _4: R.Tuple = R.memory.kill_tensor(gv6)
            gv7: R.Object = gv1[0]
            _5: R.Tuple = R.memory.kill_storage(gv7)
            gv8: R.Object = gv1[1]
            _6: R.Tuple = R.memory.kill_storage(gv8)
            return alloc2

    # fmt: on
    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
