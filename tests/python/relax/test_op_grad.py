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
from tvm.ir import Op
from tvm.script import relax as R


def test_op_correctness():
    g = relax.Var("g", R.Tensor((3, 10, 10), "float32"))
    x = relax.Var("x", R.Tensor((3, 5, 10, 10), "float32"))
    y = relax.Var("y", R.Tensor((3, 10, 10), "int64"))
    w = relax.Var("w", R.Tensor((5,), "float32"))
    assert relax.op.grad.nll_loss_backward(g, x, y, w).op == Op.get("relax.grad.nll_loss_backward")

    g = relax.Var("g", R.Tensor((3, 3, 8, 8), "float32"))
    x = relax.Var("x", R.Tensor((3, 2, 10, 10), "float32"))
    assert relax.op.grad.max_pool2d_backward(g, x, (3, 3)).op == Op.get(
        "relax.grad.max_pool2d_backward"
    )


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_nll_loss_backward_infer_struct_info():
    bb = relax.BlockBuilder()

    g = relax.Var("g", R.Tensor((3, 10, 10)))
    x = relax.Var("x", R.Tensor((3, 5, 10, 10), "float32"))
    y = relax.Var("y", R.Tensor((3, 10, 10), "int64"))
    w = relax.Var("w", R.Tensor((5,), "float32"))

    _check_inference(bb, relax.op.grad.nll_loss_backward(g, x, y), x.struct_info)
    _check_inference(bb, relax.op.grad.nll_loss_backward(g, x, y, w), x.struct_info)


def test_max_pool2d_backward_infer_struct_info():
    bb = relax.BlockBuilder()

    g = relax.Var("g", R.Tensor((3, 3, 8, 8), "float32"))
    x = relax.Var("x", R.Tensor((3, 2, 10, 10), "float32"))

    _check_inference(bb, relax.op.grad.max_pool2d_backward(g, x, (2, 2)), x.struct_info)
    _check_inference(bb, relax.op.grad.max_pool2d_backward(g, x, (3, 3)), x.struct_info)


if __name__ == "__main__":
    tvm.testing.main()
