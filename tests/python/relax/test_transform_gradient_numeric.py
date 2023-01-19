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
import numpy as np
import tvm
import tvm.testing
from tvm import relax
from tvm.relay.testing import rand
from tvm.testing import assert_allclose
from tvm.testing.utils import check_numerical_grads
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.transform import LegalizeOps


def _legalize_and_build(mod):
    lowered_mod = LegalizeOps()(mod)
    ex = relax.vm.build(lowered_mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm


def test_manual_gradient():
    # The expression computed is sum((2x - 2y) * (y + z))
    # the gradient of x is broadcast_to(2y + 2z, x.shape)
    # the gradient of y is collapse_sum_to((2x - 4y - 2z), y.shape)
    # the gradient of z is collapse_sum_to((2x - 2y), z.shape)
    # the gradient of u is 0
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 5), "float32"),
            y: R.Tensor((5,), "float32"),
            z: R.Tensor((5,), "float32"),
            u: R.Tensor((5,), "float32"),
        ):
            with R.dataflow():
                lv1 = R.add(x, x)
                lv2 = R.subtract(lv1, y)
                lv3 = R.subtract(lv2, y)
                lv4 = R.add(y, z)
                lv5 = R.multiply(lv3, lv4)
                lv6 = R.sum(lv5)
                R.output(lv6)
            return lv6

    After = relax.transform.Gradient(Before.get_global_var("main"))(Before)

    args = [rand("float32", 3, 5), rand("float32", 5), rand("float32", 5), rand("float32", 5)]
    args_np = [x.numpy() for x in args]

    vm = _legalize_and_build(After)
    output, grads = vm["main_adjoint"](*args)
    output_np = np.sum((2 * args_np[0] - 2 * args_np[1]) * (args_np[1] + args_np[2]))
    assert_allclose(output.numpy(), output_np, atol=1e-4)

    expected_grads_nd = [
        (2 * args_np[1] + 2 * args_np[2]) * np.ones_like(args_np[0]),
        np.sum((2 * args_np[0] - 4 * args_np[1] - 2 * args_np[2]), axis=0),
        np.sum((2 * args_np[0] - 2 * args_np[1]), axis=0),
        np.zeros_like(args_np[3]),
    ]
    for i, j in zip(grads, expected_grads_nd):
        assert_allclose(i.numpy(), j, atol=1e-4)


def test_mlp_blockbuilder():
    layers, in_size, out_size, hidden_size, batch_size = 3, 5, 5, 5, 4

    input_list = [relax.Var("x", R.Tensor((batch_size, in_size), "float32"))]
    w_list = (
        [relax.Var("w_0", R.Tensor((in_size, hidden_size), "float32"))]
        + [
            relax.Var("w_" + str(i + 1), R.Tensor((hidden_size, hidden_size), "float32"))
            for i in range(layers - 2)
        ]
        + [relax.Var("w_" + str(layers - 1), R.Tensor((hidden_size, out_size), "float32"))]
    )
    b_list = [
        relax.Var("b_" + str(i), R.Tensor((hidden_size,), "float32")) for i in range(layers - 1)
    ] + [relax.Var("b_" + str(layers - 1), R.Tensor((out_size,), "float32"))]
    label_list = [relax.Var("y", R.Tensor((batch_size, out_size), "float32"))]
    args_list = input_list + w_list + b_list + label_list

    bb = relax.BlockBuilder()
    with bb.function("MLP", args_list):
        with bb.dataflow():
            current = input_list[0]
            for i in range(layers):
                lv0 = bb.emit(R.matmul(current, w_list[i]))
                lv1 = bb.emit(R.add(lv0, b_list[i]))
                current = bb.emit(R.nn.relu(lv1) if i < layers - 1 else lv1)
            logits = R.nn.log_softmax(current)
            loss = bb.emit(R.nn.cross_entropy_with_logits(logits, label_list[0]))
            gv0 = bb.emit_output(loss)
        bb.emit_func_output(gv0)

    Before = bb.get()
    After = relax.transform.Gradient(Before.get_global_var("MLP"), args_list)(Before)
    # Check numerical gradients equal
    args = []
    for arg in After["MLP_adjoint"].params[:-1]:
        shape = [int(l) for l in arg.struct_info.shape]
        args.append(rand("float32", *shape))
    label = np.random.rand(batch_size, out_size).astype(np.float32)
    label /= label.sum(axis=1, keepdims=True)
    args.append(tvm.nd.array(label))

    vm_before = _legalize_and_build(Before)
    vm_after = _legalize_and_build(After)
    _, grad = vm_after["MLP_adjoint"](*args)

    def func(*inputs):
        loss = vm_before["MLP"](*[tvm.nd.array(i) for i in inputs])
        return loss.numpy()

    check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])


if __name__ == "__main__":
    tvm.testing.main()
