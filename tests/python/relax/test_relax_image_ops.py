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
import pytest
import tvm
from tvm import relax
from tvm.error import DiagnosticError
from tvm.relax.testing import transform
from tvm.script import relax as R
import tvm.testing


def test_resize2d():
    @R.function
    def expected(x: R.Tensor((2, 14, 14, 3), "float32")) -> R.Tensor(None, "float32", ndim=4):
        gv: R.Tensor((2, 28, 28, 3), "float32") = R.image.resize2d(x, size=[28, 28], layout="NHWC")
        return gv

    bb = relax.BlockBuilder()
    x = relax.Var("x", (2, 14, 14, 3), relax.DynTensorType(4, "float32"))
    with bb.function("main", [x]):
        gv = bb.emit(relax.op.image.resize2d(x, (28, 28), layout="NHWC"))
        bb.emit_func_output(gv)

    expected = expected.with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(bb.get()["main"], expected)


if __name__ == "__main__":
    test_resize2d()
