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
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op
from tvm.script import relax as R

import numpy as np


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    c = relax.Constant(tvm.nd.array(np.array([1, 2, 3], dtype="float16")))
    assert relax.op.cast(x, "float16").op == Op.get("relax.cast")
    assert relax.op.wrap_param(c, "float32").op == Op.get("relax.wrap_param")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_cast_infer_struct_info():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()

    _check_inference(bb, relax.op.cast(x, "float16"), relax.TensorStructInfo((2, 3), "float16"))


if __name__ == "__main__":
    tvm.testing.main()
