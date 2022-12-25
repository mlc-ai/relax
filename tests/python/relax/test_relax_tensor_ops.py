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
import tvm.testing
from tvm import relax
from tvm.script import relax as R


def test_add():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.add(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.add(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_subtract():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.subtract(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.subtract(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_multiply():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.multiply(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.multiply(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_divide():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.divide(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.divide(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_floor_divide():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.floor_divide(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.floor_divide(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_negative():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.negative(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.negative(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sin():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.sin(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.sin(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_cos():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.cos(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.cos(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_tanh():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.tanh(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.tanh(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sqrt():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.sqrt(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.sqrt(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_log():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.log(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.log(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_sigmoid():
    @R.function
    def expected(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.sigmoid(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.sigmoid(x))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_less():
    @R.function
    def expected(
        x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")
    ) -> R.Tensor((2, 3), "bool"):
        gv: R.Tensor((2, 3), "bool") = R.less(x, y)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 1), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.less(x, y))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


def test_ewise_fma():
    @R.function
    def expected(
        x: R.Tensor((2, 3, 4), dtype="float32"),
        y: R.Tensor((2, 3, 4), dtype="float32"),
        z: R.Tensor((2, 3, 4), dtype="float32"),
    ) -> R.Tensor((2, 3, 4), dtype="float32"):
        gv: R.Tensor((2, 3, 4), dtype="float32") = R.ewise_fma(x, y, z)
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))
    z = relax.Var("z", R.Tensor((2, 3, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y, z]):
        gv = bb.emit(relax.op.ewise_fma(x, y, z))
        bb.emit_func_output(gv)

    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
