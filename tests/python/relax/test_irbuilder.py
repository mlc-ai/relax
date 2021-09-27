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
from tvm import tir
from tvm import relax as rx


def test_dataflow_block():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.IRBuilder()

    with ib.dataflow() as df:
        lv0 = ib.emit(rx.op.add(x, y))

        assert lv0.name_hint == "lv0"
        assert lv0.shape[0] == m
        assert lv0.shape[1] == n
        assert lv0.checked_type.rank == 2
        assert lv0.checked_type.dtype == "float16"

        lv1 = ib.emit(rx.op.multiply(lv0, y))
        assert lv1.name_hint == "lv1"
        gv0 = ib.emit_output(lv1)

        assert gv0.name_hint == "gv0"
        assert gv0.shape[0] == m
        assert gv0.shape[1] == n
        assert gv0.checked_type.rank == 2
        assert gv0.checked_type.dtype == "float16"

    blocks = ib.get_blocks()
    assert len(blocks) == 1
    assert len(blocks[-1].bindings) == 3


def test_function_single_block():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.IRBuilder()

    with ib.function([x, y]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv0"
            lv1 = ib.emit(rx.op.multiply(lv0, y))
            assert lv1.name_hint == "lv1"
            gv0 = ib.emit_output(lv1)
        assert gv0.name_hint == "gv0"
        ib.emit_output(gv0)

    func = ib.get()
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.body.body == gv0
    assert gv0.shape[0] == m
    assert gv0.shape[1] == n
    assert gv0.checked_type.rank == 2
    assert gv0.checked_type.dtype == "float16"
    assert len(func.body.blocks) == 1
    assert len(func.body.blocks[0].bindings) == 3


def test_function_multi_blocks():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, n], dtype0)
    y = rx.Var("y", [n], dtype1)
    ib = rx.IRBuilder()

    with ib.function([x, y], "func"):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.name_hint == "lv0"
            gv0 = ib.emit_output(lv0)
        assert gv0.name_hint == "gv0"
        gv1 = ib.emit(rx.op.add(gv0, gv0))
        assert gv1.name_hint == "gv1"
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(gv1, gv1))
            assert lv0.name_hint == "lv0"
            gv2 = ib.emit_output(gv1)
        ib.emit_output(gv2)

    func = ib.get()
    assert gv2.shape[0] == m
    assert gv2.shape[1] == n
    assert gv2.checked_type.rank == 2
    assert gv2.checked_type.dtype == "float16"
    assert func.params[0] == x
    assert func.params[1] == y
    assert func.name.name_hint == "func"
    assert func.body.body == gv2
    assert len(func.body.blocks) == 3
    assert len(func.body.blocks[0].bindings) == 2
    assert len(func.body.blocks[1].bindings) == 1
    assert len(func.body.blocks[2].bindings) == 2


def test_binary_shape_deduction():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    k = tir.Var("k", "int32")
    dtype0 = rx.DynTensorType(rank=2, dtype="float16")
    dtype1 = rx.DynTensorType(rank=1, dtype="float16")
    x = rx.Var("x", [m, 1], dtype0)
    y = rx.Var("y", [n], dtype1)
    z = rx.Var("z", [5], dtype0)
    w = rx.Var("w", [k], dtype1)
    ib = rx.IRBuilder()

    with ib.function([x, y, z, w]):
        with ib.dataflow() as df:
            lv0 = ib.emit(rx.op.add(x, y))
            assert lv0.shape[0] == m
            assert lv0.shape[1] == n

            lv1 = ib.emit(rx.op.multiply(x, z))
            assert lv1.shape[0] == m
            assert lv1.shape[1] == 5

            lv2 = ib.emit(rx.op.multiply(z, w))
            assert isinstance(lv2.shape, tvm.relay.Call)

            lv3 = ib.emit(rx.op.multiply(y, w))
            assert isinstance(lv3.shape, tvm.relay.Call)
            gv0 = ib.emit_output(lv3)
        ib.emit_output(gv0)
        assert isinstance(gv0.shape, tvm.relay.Call)


if __name__ == "__main__":
    test_dataflow_block()
    test_function_single_block()
    test_function_multi_blocks()
    test_binary_shape_deduction()