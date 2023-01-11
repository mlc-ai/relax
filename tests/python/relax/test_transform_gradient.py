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
import pytest
import tvm
import tvm.script
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm._ffi.base import TVMError
import numpy as np


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
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(R.Tensor(None, dtype="float32", ndim=2)),
        ):
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv_adjoint, (3, 3))
                R.output(gv, x_adjoint)
            return (gv, (x_adjoint,))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
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
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(R.Tensor(None, dtype="float32", ndim=2)),
        ):
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

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    After.show()
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
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(R.Tensor(None, dtype="float32", ndim=2)),
        ):
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv2_adjoint, (3, 3)
                )
                lv: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (3, 3))
                lv11: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                lv21: R.Tensor((3, 3), dtype="float32") = R.add(lv, lv11)
                lv31: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.add(lv21, lv31)
                R.output(lv3, x_adjoint)
            return (lv3, (x_adjoint,))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
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
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(R.Tensor(None, dtype="float32", ndim=2)),
        ):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, x)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, x)
                lv3: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                R.output(lv3, x_adjoint)
            return (lv3, (x_adjoint,))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_default_require_grads():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 3), "float32"),
            y: R.Tensor((3, 3), "float32"),
            z: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv1 = R.add(x, y)
                lv2 = R.add(lv1, z)
                lv3 = R.sum(lv2)
                R.output(lv3)
            return lv3

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(
            x: R.Tensor((3, 3), dtype="float32"),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(lv3)
            return lv3

        @R.function
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32"),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(
                R.Tensor(None, dtype="float32", ndim=2),
                R.Tensor(None, dtype="float32", ndim=2),
                R.Tensor(None, dtype="float32", ndim=2),
            ),
        ):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv2_adjoint, (3, 3)
                )
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv1_adjoint, (3, 3)
                )
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv1_adjoint, (3, 3)
                )
                z_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv2_adjoint, (3, 3)
                )
                R.output(lv3, x_adjoint, y_adjoint, z_adjoint)
            return (lv3, (x_adjoint, y_adjoint, z_adjoint))

    After1 = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert_structural_equal(After1["main_adjoint"], Expected1["main_adjoint"])

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((3, 3), dtype="float32"),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                R.output(lv3)
            return lv3

        @R.function
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32"),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(R.Tensor(None, dtype="float32", ndim=2)),
        ):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = R.add(lv1, z)
                lv3: R.Tensor((), dtype="float32") = R.sum(lv2, axis=None, keepdims=False)
                lv3_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv3_adjoint, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv2_adjoint, (3, 3)
                )
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv1_adjoint, (3, 3)
                )
                R.output(lv3, x_adjoint)
            return (lv3, (x_adjoint,))

    After2 = relax.transform.Gradient(
        Before.get_global_var("main"), require_grads=Before["main"].params[0]
    )(Before)
    assert_structural_equal(After2["main_adjoint"], Expected2["main_adjoint"])


def test_tuple():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")),
            y: R.Tensor((3, 3), "float32"),
            z: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv1 = (y, z)
                lv2 = x[0]
                lv3 = lv1[0]
                lv4 = R.add(lv2, lv3)
                lv5 = R.sum(lv4)
                R.output(lv5)
            return lv5

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            with R.dataflow():
                lv1: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (y, z)
                lv2: R.Tensor((3, 3), dtype="float32") = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv2, lv3)
                lv5: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
                R.output(lv5)
            return lv5

        @R.function
        def main_adjoint(
            x: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(
                R.Tuple(
                    R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2)
                ),
                R.Tensor(None, dtype="float32", ndim=2),
                R.Tensor(None, dtype="float32", ndim=2),
            ),
        ):
            with R.dataflow():
                lv1: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (y, z)
                lv2: R.Tensor((3, 3), dtype="float32") = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv2, lv3)
                lv5: R.Tensor((), dtype="float32") = R.sum(lv4, axis=None, keepdims=False)
                lv5_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv4_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv5_adjoint, (3, 3))
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv4_adjoint, (3, 3)
                )
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv4_adjoint, (3, 3)
                )
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv1_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv3_adjoint, lv)
                lv11: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                x_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv2_adjoint, lv11)
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv3_adjoint
                z_adjoint: R.Tensor((3, 3), dtype="float32") = lv
                R.output(lv5, x_adjoint, y_adjoint, z_adjoint)
            return (lv5, (x_adjoint, y_adjoint, z_adjoint))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_tuple_assignment():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 3), "float32"),
            y: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv1 = (x, y)
                lv4 = lv1[0]
                lv7 = R.add(lv4, x)
                lv2 = lv1
                lv3 = lv2[0]
                lv5 = R.add(lv3, lv7)
                lv6 = R.sum(lv5)
                R.output(lv6)
            return lv6

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (x, y)
                lv4: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv7: R.Tensor((3, 3), dtype="float32") = R.add(lv4, x)
                lv2: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv1
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv5: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv7)
                lv6: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                R.output(lv6)
            return lv6

        @R.function
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(
                R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2)
            ),
        ):
            # block 0
            with R.dataflow():
                lv1: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (x, y)
                lv4: R.Tensor((3, 3), dtype="float32") = lv1[0]
                lv7: R.Tensor((3, 3), dtype="float32") = R.add(lv4, x)
                lv2: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv1
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv5: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv7)
                lv6: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                lv6_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv5_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv6_adjoint, (3, 3))
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv5_adjoint, (3, 3)
                )
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv2_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv3_adjoint, lv)
                lv7_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv5_adjoint, (3, 3)
                )
                lv4_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv7_adjoint, (3, 3)
                )
                lv11: R.Tensor((3, 3), dtype="float32") = R.add(lv3_adjoint, lv4_adjoint)
                lv1_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv11, lv)
                lv21: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv7_adjoint, (3, 3))
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.add(lv21, lv11)
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv
                R.output(lv6, x_adjoint, y_adjoint)
            return (lv6, (x_adjoint, y_adjoint))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_tuple_nested():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tuple(
                R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")),
                R.Tensor((3, 3), "float32"),
            ),
            y: R.Tensor((3, 3), "float32"),
            z: R.Tensor((3, 3), "float32"),
            u: R.Tensor((3, 3), "float32"),
        ):
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
        def main(
            x: R.Tuple(
                R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                R.Tensor((3, 3), dtype="float32"),
            ),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
            u: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tuple(
                    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                    R.Tensor((3, 3), dtype="float32"),
                ) = ((y, z), u)
                lv2: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv4: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv1[0]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv5)
                lv7: R.Tensor((3, 3), dtype="float32") = x[1]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv6, lv7)
                lv9: R.Tensor((), dtype="float32") = R.sum(lv8, axis=None, keepdims=False)
                R.output(lv9)
            return lv9

        @R.function
        def main_adjoint(
            x: R.Tuple(
                R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                R.Tensor((3, 3), dtype="float32"),
            ),
            y: R.Tensor((3, 3), dtype="float32"),
            z: R.Tensor((3, 3), dtype="float32"),
            u: R.Tensor((3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(
                R.Tuple(
                    R.Tuple(
                        R.Tensor(None, dtype="float32", ndim=2),
                        R.Tensor(None, dtype="float32", ndim=2),
                    ),
                    R.Tensor(None, dtype="float32", ndim=2),
                ),
                R.Tensor(None, dtype="float32", ndim=2),
                R.Tensor(None, dtype="float32", ndim=2),
                R.Tensor(None, dtype="float32", ndim=2),
            ),
        ):
            # block 0
            with R.dataflow():
                lv1: R.Tuple(
                    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                    R.Tensor((3, 3), dtype="float32"),
                ) = ((y, z), u)
                lv2: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = x[0]
                lv3: R.Tensor((3, 3), dtype="float32") = lv2[0]
                lv4: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv1[0]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.add(lv3, lv5)
                lv7: R.Tensor((3, 3), dtype="float32") = x[1]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv6, lv7)
                lv9: R.Tensor((), dtype="float32") = R.sum(lv8, axis=None, keepdims=False)
                lv9_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv8_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv9_adjoint, (3, 3))
                lv7_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv8_adjoint, (3, 3)
                )
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv8_adjoint, (3, 3)
                )
                lv5_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv6_adjoint, (3, 3)
                )
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv4_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv, lv5_adjoint)
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv6_adjoint, (3, 3)
                )
                lv11: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv2_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv3_adjoint, lv11)
                lv21: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv1_adjoint: R.Tuple(
                    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                    R.Tensor((3, 3), dtype="float32"),
                ) = ((lv, lv5_adjoint), lv21)
                x_adjoint: R.Tuple(
                    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                    R.Tensor((3, 3), dtype="float32"),
                ) = ((lv3_adjoint, lv11), lv7_adjoint)
                y_adjoint: R.Tensor((3, 3), dtype="float32") = lv
                z_adjoint: R.Tensor((3, 3), dtype="float32") = lv5_adjoint
                u_adjoint: R.Tensor((3, 3), dtype="float32") = lv21
                R.output(lv9, x_adjoint, y_adjoint, z_adjoint, u_adjoint)
            return (lv9, (x_adjoint, y_adjoint, z_adjoint, u_adjoint))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_tuple_update():
    """One tensor `x` is used in and out of tuple many times."""

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv0 = (x, y)
                lv1 = R.add(x, y)
                lv2 = lv0[0]
                lv3 = R.add(lv2, y)
                lv4 = R.add(lv1, lv3)
                lv5 = (x, y)
                lv6 = lv5[0]
                lv7 = lv0[0]
                lv8 = R.add(lv4, lv6)
                lv9 = R.add(lv8, lv7)
                lv10 = R.sum(lv9)
                R.output(lv10)
            return lv10

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv0: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (x, y)
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, y)
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv1, lv3)
                lv5: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (x, y)
                lv6: R.Tensor((3, 3), dtype="float32") = lv5[0]
                lv7: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv4, lv6)
                lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv7)
                lv10: R.Tensor((), dtype="float32") = R.sum(lv9, axis=None, keepdims=False)
                R.output(lv10)
            return lv10

        @R.function
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(
                R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2)
            ),
        ):
            # block 0
            with R.dataflow():
                lv0: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (x, y)
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv3: R.Tensor((3, 3), dtype="float32") = R.add(lv2, y)
                lv4: R.Tensor((3, 3), dtype="float32") = R.add(lv1, lv3)
                lv5: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (x, y)
                lv6: R.Tensor((3, 3), dtype="float32") = lv5[0]
                lv7: R.Tensor((3, 3), dtype="float32") = lv0[0]
                lv8: R.Tensor((3, 3), dtype="float32") = R.add(lv4, lv6)
                lv9: R.Tensor((3, 3), dtype="float32") = R.add(lv8, lv7)
                lv10: R.Tensor((), dtype="float32") = R.sum(lv9, axis=None, keepdims=False)
                lv10_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv9_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(
                    lv10_adjoint, (3, 3)
                )
                lv8_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv9_adjoint, (3, 3)
                )
                lv7_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv9_adjoint, (3, 3)
                )
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv8_adjoint, (3, 3)
                )
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv5_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv6_adjoint, lv)
                lv4_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv8_adjoint, (3, 3)
                )
                lv3_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv4_adjoint, (3, 3)
                )
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv3_adjoint, (3, 3)
                )
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv4_adjoint, (3, 3)
                )
                lv11: R.Tensor((3, 3), dtype="float32") = R.add(lv7_adjoint, lv2_adjoint)
                lv21: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv0_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv11, lv21)
                lv31: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                lv41: R.Tensor((3, 3), dtype="float32") = R.add(lv6_adjoint, lv31)
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.add(lv41, lv11)
                lv51: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv3_adjoint, (3, 3))
                lv61: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (3, 3))
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.add(lv51, lv61)
                R.output(lv10, x_adjoint, y_adjoint)
            return (lv10, (x_adjoint, y_adjoint))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_const():
    """const could be used in variable assignment, call argument, and as a part of tuple"""
    cst = relax.const(np.ones((3, 3)), dtype="float32")

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 3), "float32"),
            y: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv1 = R.add(x, cst)
                lv2 = cst
                lv3 = (cst, (cst, lv1))
                lv4 = lv3[1]
                lv5 = lv4[1]
                lv6 = R.subtract(lv5, lv2)
                gv0 = R.sum(lv6)
                R.output(gv0)
            return gv0

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, cst)
                lv2: R.Tensor((3, 3), dtype="float32") = cst
                lv3: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"),
                    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                ) = (cst, (cst, lv1))
                lv4: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv3[1]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.subtract(lv5, lv2)
                gv0: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                R.output(gv0)
            return gv0

        @R.function
        def main_adjoint(
            x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3, 3), dtype="float32")
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(
                R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2)
            ),
        ):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((3, 3), dtype="float32") = R.add(x, cst)
                lv2: R.Tensor((3, 3), dtype="float32") = cst
                lv3: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"),
                    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                ) = (cst, (cst, lv1))
                lv4: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = lv3[1]
                lv5: R.Tensor((3, 3), dtype="float32") = lv4[1]
                lv6: R.Tensor((3, 3), dtype="float32") = R.subtract(lv5, lv2)
                gv0: R.Tensor((), dtype="float32") = R.sum(lv6, axis=None, keepdims=False)
                gv0_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv6_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(gv0_adjoint, (3, 3))
                lv5_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv6_adjoint, (3, 3)
                )
                lv: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv4_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv, lv5_adjoint)
                lv11: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                lv3_adjoint: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"),
                    R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")),
                ) = (lv11, (lv, lv5_adjoint))
                lv21: R.Tensor((3, 3), dtype="float32") = R.negative(lv6_adjoint)
                lv2_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv21, (3, 3))
                lv1_adjoint: R.Tensor((3, 3), dtype="float32") = lv5_adjoint
                x_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(
                    lv1_adjoint, (3, 3)
                )
                y_adjoint: R.Tensor((3, 3), dtype="float32") = R.zeros((3, 3), dtype="float32")
                R.output(gv0, x_adjoint, y_adjoint)
            return (gv0, (x_adjoint, y_adjoint))

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_params_copy():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x0: R.Tensor((3, 3), "float32"),
            x1: R.Tensor((3, 3), "float32"),
            x2: R.Tensor((3, 3), "float32"),
            x3: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                out = R.sum(lv2)
                R.output(out)
            return out

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)
    assert len(Before["main"].params) == len(After["main"].params)
    assert len(Before["main"].params) == len(After["main_adjoint"].params)
    for i in range(len(After["main"].params)):
        assert Before["main"].params[i] == After["main"].params[i]
        assert Before["main"].params[i] != After["main_adjoint"].params[i]


def test_function_copy():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x0: R.Tensor((3, 3), "float32"),
            x1: R.Tensor((3, 3), "float32"),
            x2: R.Tensor((3, 3), "float32"),
            x3: R.Tensor((3, 3), "float32"),
        ):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                out = R.sum(lv2)
                R.output(out)
            return out

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)

    # After should have the same "main" function as Before
    assert_structural_equal(Before["main"], After["main"])

    # the first bindings of After["main_adjoint"] should be the same as Before["main"]
    old_bindings = Before["main"].body.blocks[0].bindings
    old_bindings_len = len(old_bindings)
    new_bindings = After["main_adjoint"].body.blocks[0].bindings[:old_bindings_len]
    assert_structural_equal(old_bindings, new_bindings, True)


def test_report_error():
    @I.ir_module
    class TargetNotScalar:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"), x1: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                out = R.add(x0, x1)
                R.output(out)
            return out

    with pytest.raises(TVMError):
        relax.transform.Gradient(TargetNotScalar.get_global_var("main"))(TargetNotScalar)

    @I.ir_module
    class NoDataflow:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32")):
            out = R.sum(x0)
            return out

    with pytest.raises(TVMError):
        relax.transform.Gradient(NoDataflow.get_global_var("main"))(NoDataflow)

    @I.ir_module
    class MultiBlocks:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"), x1: R.Tensor((3, 3), "float32")):
            # block 0
            with R.dataflow():
                out = R.add(x0, x1)
                R.output(out)
            # block 1
            out1 = R.sum(x0)
            return out1

    with pytest.raises(TVMError):
        relax.transform.Gradient(MultiBlocks.get_global_var("main"))(MultiBlocks)

    @I.ir_module
    class NormalModule:
        @R.function
        def main(x0: R.Tensor((3, 3), "float32"), x1: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                out = R.sum(x0)
                R.output(out)
            return out

    main_gv = NormalModule.get_global_var("main")
    # no such function
    with pytest.raises(TVMError):
        relax.transform.Gradient(MultiBlocks.get_global_var("main"))(NormalModule)
    # no such var
    with pytest.raises(TVMError):
        relax.transform.Gradient(main_gv, require_grads=MultiBlocks["main"].params[0])(NormalModule)

    @I.ir_module
    class IntDtype:
        @R.function
        def main(x: R.Tensor((3, 3), "int64")):
            with R.dataflow():
                lv1 = R.add(x, x)
                gv = R.sum(lv1)
                R.output(gv)
            return gv

    with pytest.raises(TVMError):
        relax.transform.Gradient(IntDtype.get_global_var("main"))(IntDtype)

    @I.ir_module
    class IntDtypeTuple:
        @R.function
        def main(x: R.Tuple(R.Tensor((3, 3), "int64"), R.Tensor((3, 3), "int64"))):
            with R.dataflow():
                lv1 = x[0]
                lv2 = x[1]
                lv3 = R.add(lv1, lv2)
                gv = R.sum(lv3)
                R.output(gv)
            return gv

    with pytest.raises(TVMError):
        relax.transform.Gradient(IntDtypeTuple.get_global_var("main"))(IntDtypeTuple)

    # @I.ir_module
    # class UndefinedGradient:
    #     @R.function
    #     def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3,), "int64")):
    #         with R.dataflow():
    #             gv = R.nll_loss(x, y)
    #             R.output(gv)
    #         return gv

    # with pytest.raises(TVMError):
    #     relax.transform.Gradient(UndefinedGradient.get_global_var("main"))(UndefinedGradient)


def test_mlp_script():
    """
    An example of single layer multi-layer perceptron. You can add extra layers if you want.

    For n-layer perceptron, see test_transform_gradient_numeric.py.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 10), "float32"),
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
            label: R.Tensor((3, 5), "float32"),
        ):
            with R.dataflow():
                lv0 = R.nn.matmul(x, w0)
                out = R.add(lv0, b0)
                loss = R.nn.softmax_cross_entropy(out, label)
                R.output(loss)
            return loss

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 10), dtype="float32"),
            w0: R.Tensor((10, 5), dtype="float32"),
            b0: R.Tensor((5,), dtype="float32"),
            label: R.Tensor((3, 5), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((3, 5), dtype="float32") = R.nn.matmul(x, w0, out_dtype="")
                out: R.Tensor((3, 5), dtype="float32") = R.add(lv0, b0)
                loss: R.Tensor((), dtype="float32") = R.nn.softmax_cross_entropy(out, label)
                R.output(loss)
            return loss

        @R.function
        def main_adjoint(
            x: R.Tensor((3, 10), dtype="float32"),
            w0: R.Tensor((10, 5), dtype="float32"),
            b0: R.Tensor((5,), dtype="float32"),
            label: R.Tensor((3, 5), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor(None, dtype="float32", ndim=0),
            R.Tuple(
                R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=1)
            ),
        ):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((3, 5), dtype="float32") = R.nn.matmul(x, w0, out_dtype="")
                out: R.Tensor((3, 5), dtype="float32") = R.add(lv0, b0)
                loss: R.Tensor((), dtype="float32") = R.nn.softmax_cross_entropy(out, label)
                loss_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv: R.Tensor((3, 5), dtype="float32") = R.nn.softmax(out, axis=-1)
                lv1: R.Tensor((3, 5), dtype="float32") = R.subtract(lv, label)
                out_adjoint: R.Tensor((3, 5), dtype="float32") = R.multiply(loss_adjoint, lv1)
                lv0_adjoint: R.Tensor((3, 5), dtype="float32") = R.collapse_sum_to(
                    out_adjoint, (3, 5)
                )
                lv2: R.Tensor((10, 3), dtype="float32") = R.transpose(x, axes=[1, 0])
                lv3: R.Tensor((10, 5), dtype="float32") = R.nn.matmul(
                    lv2, lv0_adjoint, out_dtype=""
                )
                w0_adjoint: R.Tensor((10, 5), dtype="float32") = R.collapse_sum_to(lv3, (10, 5))
                b0_adjoint: R.Tensor((5,), dtype="float32") = R.collapse_sum_to(out_adjoint, (5,))
                R.output(loss, w0_adjoint, b0_adjoint)
            return (loss, (w0_adjoint, b0_adjoint))

    After = relax.transform.Gradient(
        Before.get_global_var("main"), require_grads=Before["main"].params[1:3]
    )(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


if __name__ == "__main__":
    pytest.main([__file__])
