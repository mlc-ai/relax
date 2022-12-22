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
from __future__ import annotations  # must import to defer parsing of annotations

import numpy as np
import pytest
import tvm
import tvm.script
from tvm import relax
from tvm import relax as rx
from tvm.ir.base import assert_structural_equal
from tvm.relay.testing import rand
from tvm.testing import assert_allclose
from tvm.testing.utils import check_numerical_grads
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm._ffi.base import TVMError
from tvm.relax.transform import OperatorLegalizer


def _execute_mod(mod, func_name, *args):
    lowered_mod = OperatorLegalizer(mod).transform()
    ex = relax.vm.build(lowered_mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*args)


def test_simple():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                gv = R.sum(x)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                R.output(gv)
            return gv

        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2))):
            R.func_attr({"global_symbol": "main_adjoint"})
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, (3, 3))
                R.output(gv, x_adjoint)
            return (gv, (x_adjoint,))

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_assign_binding():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = x
                lv2 = lv1
                lv3 = R.sum(lv2)
                R.output(lv3)
            return lv3

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = x
                lv2: R.Tensor((3, 3), dtype="float32") = lv1
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(lv3)
            return lv3

        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = x
                lv2: R.Tensor((3, 3), dtype="float32") = lv1
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv2_adjoint
                x_adjoint: R.Tensor((3, 3), dtype="float32") = lv1_adjoint
                R.output(lv3, x_adjoint)
            return (lv3, (x_adjoint,))

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_multiple_uses():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.add(x, x)
                lv2 = R.add(lv1, x)
                lv3 = R.sum(lv2)
                R.output(lv3)
            return lv3

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(lv3)
            return lv3

        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2))):
            R.func_attr({"global_symbol": "main_adjoint"})
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (3, 3))
                lv: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (3, 3))
                lv11: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                lv21: R.Tensor((3, 3), dtype="float32") = R.add(lv, lv11)
                lv31: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.add(lv21, lv31)
                R.output(lv3, x_adjoint)
            return (lv3, (x_adjoint,))

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_unused():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.add(x, x)
                lv2 = R.add(lv1, x)
                lv3 = R.sum(x)
                R.output(lv3)
            return lv3

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                lv3: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                R.output(lv3)
            return lv3

        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                lv3: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                R.output(lv3, x_adjoint)
            return (lv3, (x_adjoint,))

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_default_require_grads():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"),
                 y: R.Tensor((3, 3), "float32"),
                 z: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.add(x, y)
                lv2 = R.add(lv1, z)
                lv3 = R.sum(lv2)
                R.output(lv3)
            return lv3

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(lv3)
            return lv3

        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (3, 3))
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                z_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (3, 3))
                R.output(lv3, x_adjoint, y_adjoint, z_adjoint)
            return (lv3, (x_adjoint, y_adjoint, z_adjoint))

    After1 = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After1["main_adjoint"], Expected1["main_adjoint"])

    @I.ir_module
    class Expected2:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(lv3)
            return lv3

        @R.function
        def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (3, 3))
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                z_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (3, 3))
                R.output(lv3, x_adjoint, y_adjoint, z_adjoint)
            return (lv3, (x_adjoint, y_adjoint, z_adjoint))

    After2 = relax.transform.SimpleAD(Before.get_global_var("main"), require_grads=Before["main"].params[0])(Before)
    assert_structural_equal(After2["main_adjoint"], Expected2["main_adjoint"])


def test_tuple():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")),
                 y: R.Tensor((3, 3), "float32"),
                 z: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.Tuple((y, z))
                lv2 = x[0]
                lv3 = lv1[0]
                lv4 = R.add(lv2, lv3)
                lv5 = R.sum(lv4)
                R.output(lv5)
            return lv5

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (y, z)
                lv2: R.Tensor((3, 3), dtype="float32") = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv2, lv3)
                lv5: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
                R.output(lv5)
            return lv5

        @R.function
        def main_adjoint(x: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tuple(R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2)), R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2))):
            R.func_attr({"global_symbol": "main_adjoint"})
            with R.dataflow():
                lv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (y, z)
                lv2: R.Tensor((3, 3), dtype="float32") = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv2, lv3)
                lv5: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
                lv5_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv4_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv5_adjoint, (3, 3))
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv4_adjoint, (3, 3))
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv4_adjoint, (3, 3))
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv1_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv3_adjoint, lv)
                lv11: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                x_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv2_adjoint, lv11)
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv3_adjoint
                z_adjoint: R.Tensor((3, 3), dtype="float32") = lv
                R.output(lv5, x_adjoint, y_adjoint, z_adjoint)
            return (lv5, (x_adjoint, y_adjoint, z_adjoint))

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])
    # Expected.show()

test_tuple()
def test_tuple_nested():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tuple(R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")),
                            R.Tensor((3, 3), "float32")),
                 y: R.Tensor((3, 3), "float32"),
                 z: R.Tensor((3, 3), "float32"),
                 u: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = ((y, z), u)
                lv2 = x[0]
                lv3 = lv2[0]
                lv4 = lv1[0]
                lv5 = lv4[1]
                lv6 = R.add(lv3, lv5)
                lv7 = x[1]
                lv8 = R.add(lv6, lv7)
                lv9 = R.sum(lv8)
                R.output(lv9)
            return lv9

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32"), u: R.Tensor((3, 3), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((y, z), u)
                lv2: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1[0]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv5)
                lv7: R.Tensor((3, 3), dtype="float32") = x[1]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv6, lv7)
                lv9: R.Tensor((), dtype="float32") = R.sum(lv8, axis=None, keepdims=False)
                R.output(lv9)
            return lv9

        @R.function
        def main_adjoint(x: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")), y: R.Tensor((3, 3), dtype="float32"), z: R.Tensor((3, 3), dtype="float32"), u: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tuple(R.Tuple(R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2)), R.Tensor(None, dtype="float32", ndim=2)), R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv1: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((y, z), u)
                lv2: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv4: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = lv1[0]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv5)
                lv7: R.Tensor((3, 3), dtype="float32") = x[1]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv6, lv7)
                lv9: R.Tensor((), dtype="float32") = R.sum(lv8, axis=None, keepdims=False)
                lv9_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv8_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv9_adjoint, (3, 3))
                lv7_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv8_adjoint, (3, 3))
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv8_adjoint, (3, 3))
                lv5_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv6_adjoint, (3, 3))
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv4_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv, lv5_adjoint)
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv6_adjoint, (3, 3))
                lv11: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv2_adjoint: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")) = (lv3_adjoint, lv11)
                lv21: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv1_adjoint: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((lv, lv5_adjoint), lv21)
                x_adjoint: R.Tuple(R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")), R.Tensor((3, 3), dtype="float32")) = ((lv3_adjoint, lv11), lv7_adjoint)
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv
                z_adjoint: R.Tensor((3, 3), dtype="float32") = lv5_adjoint
                u_adjoint: R.Tensor((3, 3), dtype="float32") = lv21
                R.output(lv9, x_adjoint, y_adjoint, z_adjoint, u_adjoint)
            return (lv9, (x_adjoint, y_adjoint, z_adjoint, u_adjoint))

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])

# test_tuple_nested()
# tuple多次更新
# 建新tuple
def test_tuple_complex():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"),
                 y: R.Tensor((3, 3), "float32"),
                 z: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv0 = (x, y)

                R.output(z9)
            return z9

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)

    x1 = rand("float32", *(3, 3))
    x2 = rand("float32", *(3, 3))
    y = rand("float32", *(3, 3))
    args_numpy = [x1.numpy(), x2.numpy(), y.numpy()]

    _, grad = _execute_mod(After, "main_adjoint", x1, x2, y)

    def func(*inputs):
        loss = _execute_mod(Before, "main", *[tvm.nd.array(i) for i in inputs])
        return loss.numpy()

    check_numerical_grads(func, args_numpy, [i.numpy() for i in grad])


def test_params_copy():
    @I.ir_module
    class Before:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"),
                 x1: R.Tensor((3, 3), "float32"),
                 x2: R.Tensor((3, 3), "float32"),
                 x3: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                out = R.sum(lv2)
                R.output(out)
            return out

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert len(Before["main"].params) == len(After["main"].params)
    assert len(Before["main"].params) == len(After["main_adjoint"].params)
    for i in range(len(After["main"].params)):
        assert Before["main"].params[i] == After["main"].params[i]
        assert Before["main"].params[i] != After["main_adjoint"].params[i]


def test_function_copy():
    @I.ir_module
    class Before:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"),
                 x1: R.Tensor((3, 3), "float32"),
                 x2: R.Tensor((3, 3), "float32"),
                 x3: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                out = R.sum(lv2)
                R.output(out)
            return out

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    inputs = [rand("float32", 3, 3) for _ in range(4)]
    out1 = _execute_mod(Before, "main", *inputs)
    out2, _ = _execute_mod(After, "main_adjoint", *inputs)
    assert rx.analysis.well_formed(After)
    assert(out1.numpy() == out2.numpy())


def test_ad_error_cases():
    @I.ir_module
    class TargetNotScalar:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"),
                 x1: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                out = R.add(x0, x1)
                R.output(out)
            return out
    with pytest.raises(TVMError):
        relax.transform.SimpleAD(TargetNotScalar.get_global_var("main"))(TargetNotScalar)

    @I.ir_module
    class NoDataflow:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32")):
            out = R.sum(x0)
            return out
    with pytest.raises(TVMError):
        relax.transform.SimpleAD(NoDataflow.get_global_var("main"))(NoDataflow)

    @I.ir_module
    class MultiBlocks:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"),
                 x1: R.Tensor((3, 3), "float32")):
            # block 0
            with R.dataflow():
                out = R.add(x0, x1)
                R.output(out)
            # block 1
            out1 = R.sum(x0)
            return out1
    with pytest.raises(TVMError):
        relax.transform.SimpleAD(MultiBlocks.get_global_var("main"))(MultiBlocks)

    @I.ir_module
    class NormalModule:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"),
                x1: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                out = R.sum(x0)
                R.output(out)
            return out

    main_gv = NormalModule.get_global_var("main")
    # no such function
    with pytest.raises(TVMError):
        relax.transform.SimpleAD(MultiBlocks.get_global_var("main"))(NormalModule)
    # no such var
    with pytest.raises(TVMError):
        relax.transform.SimpleAD(main_gv, require_grads=MultiBlocks["main"].params[0])(NormalModule)




def test_mlp_script():
    """
    An example of single layer multi-layer perceptron. You can add extra layers if you want.

    For n-layer perceptron, see test_transform_gradient_numeric.py.
    """
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 10), "float32"),
                 w0: R.Tensor((10, 5), "float32"),
                 b0: R.Tensor((5,), "float32"),
                 label: R.Tensor((3, 5), "float32")):
            with R.dataflow():
                lv0 = R.nn.matmul(x, w0)
                out = R.add(lv0, b0)
                loss = R.nn.softmax_cross_entropy(out, label)
                R.output(loss)
            return loss


    After = relax.transform.SimpleAD(Before.get_global_var("main"), require_grads=Before["main"].params[1:3])(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])





# if __name__ == "__main__":
#     pytest.main([__file__])
