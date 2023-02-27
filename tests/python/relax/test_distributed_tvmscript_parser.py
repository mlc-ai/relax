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

from typing import Optional, Union


import pytest
import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax, tir, topi

from tvm.ir import Range
from tvm.relax import SeqExpr, VarBinding, Call
from tvm.relax.distributed import DeviceMesh
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Union[relax.Function, IRModule],
):
    # disable roundtrip check for now
    tvm.ir.assert_structural_equal(parsed, expect)


# move to mainline test in the future
def test_module_with_attr_and_global_info():
    @I.ir_module
    class TestModule:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "device_mesh": [
                    R.device_mesh((2, 2), R.range(0, 4)),  # mesh[0]
                    R.device_mesh((1,), R.range(4, 5)),  # mesh[1]
                ]
            }
        )

        @T.prim_func
        def tir_func(
            x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
            y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            # TODO(Siyuan): Need to change to `TestModule.tir_func`
            gv0 = R.call_tir(tir_func, x, R.Tensor((128, 128), dtype="float32"))
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, primfunc_name_hint="tir_func")
        bb.emit_func_output(out)
    mod = bb.get()
    mod.update_global_info(
        "device_mesh", [DeviceMesh((2, 2), Range(0, 4)), DeviceMesh((1,), Range(4, 5))]
    )
    mod = mod.with_attr("device_num", tvm.tir.IntImm("int32", 10))
    _check(TestModule, mod)


def test_call_tir_dtensor():
    @I.ir_module
    class TestModule:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    I.device_mesh((2, 2), I.Range(0, 4)),  # mesh[0]
                    I.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @T.prim_func
        def tir_func(
            x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
            y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with T.block():
                    vi, vj = T.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"),
        ) -> R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"):
            gv0 = R.dist.call_tir(
                tir_func,
                x,
                R.DTensor(
                    shape=(128, 128), dtype="float32", device_mesh="mesh[0]", placement="S[0], R"
                ),
            )
            return gv0

    device_mesh_list = [DeviceMesh((2, 2), Range(0, 4)), DeviceMesh((1,), Range(4, 5))]
    foo_func = TestModule["foo"]
    params = foo_func.params
    assert len(params) == 1
    assert params[0].struct_info == R.DTensor(
        (128, 128), "float32", device_mesh_list[0], placement="S[0], R"
    )
    assert foo_func.ret_struct_info == R.DTensor(
        (128, 128), "float32", device_mesh_list[0], placement="S[0], R"
    )
    assert isinstance(foo_func.body, SeqExpr)
    assert len(foo_func.body.blocks[0].bindings) == 1
    assert isinstance(foo_func.body.blocks[0].bindings[0], VarBinding)
    value = foo_func.body.blocks[0].bindings[0].value
    assert isinstance(value, Call)
    assert value.sinfo_args[0] == R.DTensor(
        (128, 128), "float32", device_mesh_list[0], placement="S[0], R"
    )
    print(TestModule.script())


if __name__ == "__main__":
    test_call_tir_dtensor()
