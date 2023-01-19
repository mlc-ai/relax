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
from typing import Callable
import tvm
from tvm import relax
from tvm.ir.op import Op
from tvm.script import relax as R, tir as T
import tvm.testing


def _map_to_grad(op_func: Callable, op_name: str, grad_var, *args):
    op = Op.get(op_name)
    op_grad_func = op.get_attr("FPrimalGradient")
    bb = relax.BlockBuilder()
    orig_call = bb.normalize(op_func(*args))
    orig_var = relax.Var("orig_var")
    relax.expr._update_struct_info(orig_var, orig_call.struct_info)
    grad_call = relax.Tuple(op_grad_func(orig_var, orig_call, grad_var, bb))
    return grad_call


def test_collapse_eliminated():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    output_grad = relax.Var("og", R.Tensor((2, 3), "float32"))
    grads = _map_to_grad(relax.op.add, "relax.add", output_grad, x, y)
    tvm.ir.assert_structural_equal(grads, relax.Tuple((output_grad, output_grad)))


def test_collapse_not_eliminated():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    output_grad = relax.Var("og", R.Tensor((2, 3), "float32"))
    grads = _map_to_grad(relax.op.add, "relax.add", output_grad, x, y)
    assert isinstance(grads[1], relax.Call) and grads[1].op == Op.get("relax.collapse_sum_to")


if __name__ == "__main__":
    tvm.testing.main()
