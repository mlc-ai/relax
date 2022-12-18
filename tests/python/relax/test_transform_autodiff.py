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


def test_binding_uses():
    # This case tests:
    # - Different type of bindings: assign binding & call binding;
    # - One def and multiple uses.
    # - Unused variable in module
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((5, 5), "float32"),
                 y: R.Tensor((5,), "float32"),
                 z: R.Tensor((5,), "float32"),
                 u: R.Tensor((5,), "float32")):
            with R.dataflow():
                lv1 = x
                lv2 = R.add(lv1, y)
                lv3 = R.add(lv2, y)
                lv4 = R.add(x, lv3)
                lv5 = lv3
                lv6 = R.add(x, lv5)
                lv7 = R.sum(lv4)
                lv8 = R.add(lv6, z) # unused
                R.output(lv7)
            return lv7
    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    args = [rand("float32", 5, 5), rand("float32", 5), rand("float32", 5), rand("float32", 5)]
    output, grads = _execute_mod(After, "main_adjoint", *args)
    assert_allclose(output.numpy(), np.sum(2 * args[0].numpy() + 2 * args[1].numpy()), atol=1e-4)
    expected_grads_nd = [2 * np.ones_like(args[0].numpy()),
                         10 * np.ones_like(args[1].numpy()),
                         np.zeros_like(args[2].numpy()),
                         np.zeros_like(args[3].numpy())]

    for i, j in zip(grads, expected_grads_nd):
        assert_allclose(i.numpy(), j)


def test_default_require_grads():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((5, 5), "float32"),
                 y: R.Tensor((5, 5), "float32"),
                 z: R.Tensor((5, 5), "float32"),
                 u: R.Tensor((5, 5), "float32")):
            with R.dataflow():
                lv1 = R.add(x, y)
                lv2 = R.subtract(z, u)
                lv3 = R.add(y, z)
                lv4 = R.add(lv1, lv2)
                lv5 = R.add(lv4, lv3)
                lv6 = R.sum(lv5)
                R.output(lv6)
            return lv6

    @tvm.script.ir_module
    class Expected1:
        @R.function
        def main(x: R.Tensor((5, 5), dtype="float32"), y: R.Tensor((5, 5), dtype="float32"), z: R.Tensor((5, 5), dtype="float32"), u: R.Tensor((5, 5), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((5, 5), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((5, 5), dtype="float32") = R.subtract(z, u)
                lv3: R.Tensor((5, 5), dtype="float32") = R.add(y, z)
                lv4: R.Tensor((5, 5), dtype="float32") = R.add(lv1, lv2)
                lv5: R.Tensor((5, 5), dtype="float32") = R.add(lv4, lv3)
                lv6: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                R.output(lv6)
            return lv6

        @R.function
        def main_adjoint(x: R.Tensor((5, 5), dtype="float32"), y: R.Tensor((5, 5), dtype="float32"), z: R.Tensor((5, 5), dtype="float32"), u: R.Tensor((5, 5), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((5, 5), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((5, 5), dtype="float32") = R.subtract(z, u)
                lv3: R.Tensor((5, 5), dtype="float32") = R.add(y, z)
                lv4: R.Tensor((5, 5), dtype="float32") = R.add(lv1, lv2)
                lv5: R.Tensor((5, 5), dtype="float32") = R.add(lv4, lv3)
                lv6: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                lv6_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv5_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(lv6_adjoint, (5, 5))
                lv4_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv5_adjoint, (5, 5))
                lv3_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv5_adjoint, (5, 5))
                lv2_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv4_adjoint, (5, 5))
                lv1_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv4_adjoint, (5, 5))
                x_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (5, 5))
                lv: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv3_adjoint, (5, 5))
                lv11: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (5, 5))
                y_adjoint: R.Tensor((5, 5), dtype="float32") = R.add(lv, lv11)
                lv21: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv3_adjoint, (5, 5))
                lv31: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv2_adjoint, (5, 5))
                z_adjoint: R.Tensor((5, 5), dtype="float32") = R.add(lv21, lv31)
                lv41: R.Tensor((5, 5), dtype="float32") = R.negative(lv2_adjoint)
                u_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv41, (5, 5))
                R.output(lv6, x_adjoint, y_adjoint, z_adjoint, u_adjoint)
            return (lv6, (x_adjoint, y_adjoint, z_adjoint, u_adjoint))



    After1 = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    assert_structural_equal(After1["main_adjoint"], Expected1["main_adjoint"])

    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(x: R.Tensor((5, 5), dtype="float32"), y: R.Tensor((5, 5), dtype="float32"), z: R.Tensor((5, 5), dtype="float32"), u: R.Tensor((5, 5), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((5, 5), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((5, 5), dtype="float32") = R.subtract(z, u)
                lv3: R.Tensor((5, 5), dtype="float32") = R.add(y, z)
                lv4: R.Tensor((5, 5), dtype="float32") = R.add(lv1, lv2)
                lv5: R.Tensor((5, 5), dtype="float32") = R.add(lv4, lv3)
                lv6: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                R.output(lv6)
            return lv6

        @R.function
        def main_adjoint(x: R.Tensor((5, 5), dtype="float32"), y: R.Tensor((5, 5), dtype="float32"), z: R.Tensor((5, 5), dtype="float32"), u: R.Tensor((5, 5), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=2))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((5, 5), dtype="float32") = R.add(x, y)
                lv2: R.Tensor((5, 5), dtype="float32") = R.subtract(z, u)
                lv3: R.Tensor((5, 5), dtype="float32") = R.add(y, z)
                lv4: R.Tensor((5, 5), dtype="float32") = R.add(lv1, lv2)
                lv5: R.Tensor((5, 5), dtype="float32") = R.add(lv4, lv3)
                lv6: R.Tensor((), dtype="float32") = R.sum(lv5, axis=None, keepdims=False)
                lv6_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv5_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(lv6_adjoint, (5, 5))
                lv4_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv5_adjoint, (5, 5))
                lv3_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv5_adjoint, (5, 5))
                lv2_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv4_adjoint, (5, 5))
                lv1_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv4_adjoint, (5, 5))
                x_adjoint: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (5, 5))
                lv: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv3_adjoint, (5, 5))
                lv11: R.Tensor((5, 5), dtype="float32") = R.collapse_sum_to(lv1_adjoint, (5, 5))
                y_adjoint: R.Tensor((5, 5), dtype="float32") = R.add(lv, lv11)
                R.output(lv6, x_adjoint, y_adjoint)
            return (lv6, (x_adjoint, y_adjoint))

    After2 = relax.transform.SimpleAD(Before.get_global_var("main"), require_grads=Before["main"].params[:2])(Before)
    assert_structural_equal(After2["main_adjoint"], Expected2["main_adjoint"])


def test_mlp_script():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((5, 20), "float32"),
                 w0: R.Tensor((20, 10), "float32"),
                 b0: R.Tensor((10,), "float32"),
                 label: R.Tensor((5, 10), "float32")):
            with R.dataflow():
                lv0 = R.nn.matmul(x, w0)
                out = R.add(lv0, b0)
                loss = R.nn.softmax_cross_entropy(out, label)
                R.output(loss)
            return loss

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 20), dtype="float32"), w0: R.Tensor((20, 10), dtype="float32"), b0: R.Tensor((10,), dtype="float32"), label: R.Tensor((5, 10), dtype="float32")) -> R.Tensor(None, dtype="float32", ndim=0):
            # block 0
            with R.dataflow():
                lv0: R.Tensor((5, 10), dtype="float32") = R.nn.matmul(x, w0, out_dtype="")
                out: R.Tensor((5, 10), dtype="float32") = R.add(lv0, b0)
                loss: R.Tensor((), dtype="float32") = R.nn.softmax_cross_entropy(out, label)
                R.output(loss)
            return loss

        @R.function
        def main_adjoint(x: R.Tensor((5, 20), dtype="float32"), w0: R.Tensor((20, 10), dtype="float32"), b0: R.Tensor((10,), dtype="float32"), label: R.Tensor((5, 10), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2), R.Tensor(None, dtype="float32", ndim=1))):
            # function attr dict
            R.func_attr({"global_symbol": "main_adjoint"})
            # block 0
            with R.dataflow():
                lv0: R.Tensor((5, 10), dtype="float32") = R.nn.matmul(x, w0, out_dtype="")
                out: R.Tensor((5, 10), dtype="float32") = R.add(lv0, b0)
                loss: R.Tensor((), dtype="float32") = R.nn.softmax_cross_entropy(out, label)
                loss_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
                lv: R.Tensor((5, 10), dtype="float32") = R.nn.softmax(out, axis=-1)
                lv1: R.Tensor((5, 10), dtype="float32") = R.subtract(lv, label)
                out_adjoint: R.Tensor((5, 10), dtype="float32") = R.multiply(loss_adjoint, lv1)
                lv0_adjoint: R.Tensor((5, 10), dtype="float32") = R.collapse_sum_to(out_adjoint, (5, 10))
                lv2: R.Tensor((20, 5), dtype="float32") = R.transpose(x, axes=[1, 0])
                lv3: R.Tensor((20, 10), dtype="float32") = R.nn.matmul(lv2, lv0_adjoint, out_dtype="")
                w0_adjoint: R.Tensor((20, 10), dtype="float32") = R.collapse_sum_to(lv3, (20, 10))
                b0_adjoint: R.Tensor((10,), dtype="float32") = R.collapse_sum_to(out_adjoint, (10,))
                R.output(loss, w0_adjoint, b0_adjoint)
            return (loss, (w0_adjoint, b0_adjoint))

    After = relax.transform.SimpleAD(Before.get_global_var("main"), require_grads=Before["main"].params[1:3])(Before)
    assert_structural_equal(After["main_adjoint"], Expected["main_adjoint"])


def test_mlp_blockbuilder():
    layers, in_size, out_size, hidden_size, batch_size = 3, 5, 5, 5, 4

    ty1 = rx.DynTensorType(ndim=1, dtype="float32")
    ty2 = rx.DynTensorType(ndim=2, dtype="float32")

    input_list = [rx.Var("x", [batch_size, in_size], ty2)]
    w_list = [rx.Var("w_0", [in_size, hidden_size], ty2)] + \
        [rx.Var("w_" + str(i + 1), [hidden_size, hidden_size], ty2) for i in range(layers - 2)] + \
        [rx.Var("w_" + str(layers - 1), [hidden_size, out_size], ty2)]
    b_list = [rx.Var("b_" + str(i), [hidden_size], ty1) for i in range(layers - 1)] + \
        [rx.Var("b_" + str(layers - 1), [out_size], ty1)]
    label_list = [rx.Var("y", [batch_size, out_size], ty2)]
    args_list = input_list + w_list + b_list + label_list

    bb = rx.BlockBuilder()
    with bb.function("MLP", args_list):
        with bb.dataflow():
            current = input_list[0]
            for i in range(layers):
                lv0 = bb.emit(R.nn.matmul(current, w_list[i]))
                lv1 = bb.emit(R.add(lv0, b_list[i]))
                current = bb.emit(R.nn.relu(lv1) if i < layers - 1 else lv1)
            loss = bb.emit(R.nn.softmax_cross_entropy(current, label_list[0]))
            gv0 = bb.emit_output(loss)
        bb.emit_func_output(gv0)

    Before = bb.get()
    After = relax.transform.SimpleAD(Before.get_global_var("MLP"), args_list)(Before)
    # Check numerical gradients equal
    args = []
    for arg in After["MLP_adjoint"].params[:-1]:
        shape = [int(l) for l in arg.shape]
        args.append(rand("float32", *shape))
    label = np.random.rand(batch_size, out_size).astype(np.float32)
    label /= label.sum(axis=1, keepdims=True)
    args.append(tvm.nd.array(label))

    _, grad = _execute_mod(After, "MLP_adjoint", *args)

    def func(*inputs):
        loss = _execute_mod(Before, "MLP", *[tvm.nd.array(i) for i in inputs])
        return loss.numpy()
    check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])


def test_gradient_api():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"),
                 y: R.Tensor((3,), "float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                out = R.sum(lv0)
                R.output(out)
            return out

    @R.function
    def main_adjoint(x: R.Tensor((3, 3), dtype="float32"), y: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor(None, dtype="float32", ndim=0), R.Tuple(R.Tensor(None, dtype="float32", ndim=2))):
        # function attr dict
        R.func_attr({"global_symbol": "main_adjoint"})
        # block 0
        with R.dataflow():
            lv0: R.Tensor((3, 3), dtype="float32") = R.add(x, y)
            out: R.Tensor((), dtype="float32") = R.sum(lv0, axis=None, keepdims=False)
            out_adjoint: R.Tensor((), dtype="float32") = R.ones((), dtype="float32")
            lv0_adjoint: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(out_adjoint, (3, 3))
            x_adjoint: R.Tensor((3, 3), dtype="float32") = R.collapse_sum_to(lv0_adjoint, (3, 3))
            R.output(out, x_adjoint)
        return (out, (x_adjoint,))

    after_func = relax.transform.gradient(Before["main"], require_grads=Before["main"].params[0])
    assert_structural_equal(after_func, main_adjoint)
    after_func = relax.transform.gradient(Before.get_global_var("main"), Before["main"].params[0],
                                          mod=Before)
    assert_structural_equal(after_func, main_adjoint)


def test_tuple1():
    @I.ir_module
    class Before:
        @R.function
        def main(x1: R.Tensor((1, 10), "float32"),
                 y1: R.Tensor((1, 10), "float32"),
                 x2: R.Tensor((1, 10), "float32"),
                 y2: R.Tensor((1, 10), "float32"),
                 z: R.Tensor((1, 10), "float32")):
            with R.dataflow():
                t1 = R.Tuple((x1, y1))
                lv1 = R.add(R.TupleGetItem(t1, 0), R.TupleGetItem(t1, 1))
                t2 = relax.Tuple((x2, y2))
                lv2 = R.subtract(R.TupleGetItem(t2, 1), lv1)
                lv3 = R.multiply(lv2, R.TupleGetItem(t2, 0))
                loss = R.nn.softmax_cross_entropy(lv3, z)
                R.output(loss)
            return loss

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)

    args = []
    for arg in After["main_adjoint"].params[:-1]:
        shape = [int(l) for l in arg.shape]
        args.append(rand("float32", *shape))

    z = np.random.rand(1, 10).astype(np.float32)
    z /= z.sum(axis=1, keepdims=True)
    args.append(tvm.nd.array(z))

    _, grad = _execute_mod(After, "main_adjoint", *args)

    def func(*inputs):
        loss = _execute_mod(Before, "main", *[tvm.nd.array(i) for i in inputs])
        return loss.numpy()

    check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])


def test_tuple2():
    @I.ir_module
    class Before:
        @R.function
        def main(x1: R.Tensor((1, 10), "float32"),
                 y1: R.Tensor((1, 10), "float32"),
                 x2: R.Tensor((1, 10), "float32"),
                 y2: R.Tensor((1, 10), "float32"),
                 z: R.Tensor((1, 10), "float32")):
            with R.dataflow():
                t = relax.Tuple( (relax.Tuple((x1, y1)), relax.Tuple((x2, y2))) )
                t0 = relax.TupleGetItem(t, 0)
                t1 = relax.TupleGetItem(t, 1)
                lv1 = R.add(relax.TupleGetItem(t0, 0), relax.TupleGetItem(t0, 1))
                lv2 = R.subtract(relax.TupleGetItem(t1, 1), lv1)
                lv3 = R.multiply(lv2, relax.TupleGetItem(t1, 0))
                loss = R.nn.softmax_cross_entropy(lv3, z)
                R.output(loss)
            return loss

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)

    args = []
    for arg in After["main_adjoint"].params[:-1]:
        shape = [int(l) for l in arg.shape]
        args.append(rand("float32", *shape))

    z = np.random.rand(1, 10).astype(np.float32)
    z /= z.sum(axis=1, keepdims=True)
    args.append(tvm.nd.array(z))

    _, grad = _execute_mod(After, "main_adjoint", *args)

    def func(*inputs):
        loss = _execute_mod(Before, "main", *[tvm.nd.array(i) for i in inputs])
        return loss.numpy()

    check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])


def test_tuple3():
    @I.ir_module
    class Before:
        @R.function
        def main(x0: R.Tensor((10, 5), "float32"),
                 x1: R.Tensor((10, 5), "float32"),
                 y: R.Tensor((10, 5), "float32")):
            with R.dataflow():
                x = relax.Tuple((x0, x1))
                z0 = relax.Tuple( (x, relax.Tuple( (x, x) )) )
                z1 = relax.TupleGetItem(z0, 1)
                z2 = relax.TupleGetItem(z1, 0)
                z3 = relax.TupleGetItem(z2, 1)
                z4 = R.multiply(z3, y)
                z10 = relax.Tuple((z3, y))
                z5 = relax.TupleGetItem(z10, 1)
                z6 = R.add(z5, z4)
                z7 = relax.TupleGetItem(x, 0)
                z8 = R.add(z7, z6)
                z9 = R.sum(z8)
                R.output(z9)
            return z9

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)

    x1 = rand("float32", *(10, 5))
    x2 = rand("float32", *(10, 5))
    y = rand("float32", *(10, 5))
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
        def main(x0: R.Tensor((5, 5), "float32"),
                 x1: R.Tensor((5, 5), "float32"),
                 x2: R.Tensor((5, 5), "float32"),
                 x3: R.Tensor((5, 5), "float32")):
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
        def main(x0: R.Tensor((5, 5), "float32"),
                 x1: R.Tensor((5, 5), "float32"),
                 x2: R.Tensor((5, 5), "float32"),
                 x3: R.Tensor((5, 5), "float32")):
            with R.dataflow():
                lv0 = R.add(x0, x1)
                lv1 = R.add(x2, x3)
                lv2 = R.add(lv0, lv1)
                out = R.sum(lv2)
                R.output(out)
            return out

    After = relax.transform.SimpleAD(Before.get_global_var("main"))(Before)
    inputs = [rand("float32", 5, 5) for _ in range(4)]
    out1 = _execute_mod(Before, "main", *inputs)
    out2, _ = _execute_mod(After, "main_adjoint", *inputs)
    assert rx.analysis.well_formed(After)
    assert(out1.numpy() == out2.numpy())


def test_ad_error_cases():
    @I.ir_module
    class TargetNotScalar:
        @R.function
        def main(x0: R.Tensor((5, 5), "float32"),
                 x1: R.Tensor((5, 5), "float32")):
            with R.dataflow():
                out = R.add(x0, x1)
                R.output(out)
            return out
    with pytest.raises(TVMError):
        relax.transform.SimpleAD(TargetNotScalar.get_global_var("main"))(TargetNotScalar)

    @I.ir_module
    class NoDataflow:
        @R.function
        def main(x0: R.Tensor((5, 5), "float32")):
            out = R.sum(x0)
            return out
    with pytest.raises(TVMError):
        relax.transform.SimpleAD(NoDataflow.get_global_var("main"))(NoDataflow)

    @I.ir_module
    class MultiBlocks:
        @R.function
        def main(x0: R.Tensor((5, 5), "float32"),
                 x1: R.Tensor((5, 5), "float32")):
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
        def main(x0: R.Tensor((5, 5), "float32"),
                x1: R.Tensor((5, 5), "float32")):
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


if __name__ == "__main__":
    pytest.main([__file__])
