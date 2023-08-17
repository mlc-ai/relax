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
import pytest
import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.topi.utils import get_const_tuple


def with_tvm(lam, shapes, ops, out_shapes):
    """Take numpy arrays as args, convert them to TVM tensors and call `lam`.
    Result of lambda is converted back to numpy array and returned.
    """
    dev = tvm.cpu(0)
    pls = []  # placeholders
    vals_nd = []  # initial values
    out_nd = []  # output values
    for i, (shape, arg) in enumerate(zip(shapes, ops)):
        pls.append(te.placeholder(shape, name="pl" + str(i)))
        vals_nd.append(tvm.nd.array(arg, dev))

    outputs = lam(*pls)
    if isinstance(outputs, tvm.ir.container.Array):
        outputs = [outputs[i] for i in range(len(outputs))]
    else:
        outputs = [outputs]
    for out_shape, out in zip(out_shapes, outputs):
        out_nd.append(tvm.nd.array(np.zeros(out_shape).astype(out.dtype), device=dev))
    func = te.create_prim_func(pls + outputs)
    m = tvm.build(func, target="llvm")
    m(*(vals_nd + out_nd))
    return [out_.numpy() for out_ in out_nd]


def verify_einsum(subscripts, shapes, fcompute, fcombine, fidentity, np_lambda, shape_dict={}):
    ops = []  # ndarrays to be used as inputs
    symbolic_shapes = []  # shapes to declare the placeholders
    name_to_var = {}

    def get_concrete_shape(shape):
        return [shape_dict[s] if isinstance(s, str) else s for s in shape]

    def get_symblic_shape_var(name, dtype="int32"):
        if name not in name_to_var:
            name_to_var[name] = te.var(name, dtype=dtype)
        return name_to_var[name]

    def get_symbolic_shape(shape):
        return [get_symblic_shape_var(s) if isinstance(s, str) else s for s in shape]

    for shape in shapes:
        concrete_shape = get_concrete_shape(shape)
        tmp = np.random.uniform(low=-1.0, high=1.0, size=concrete_shape).astype(np.float32)
        ops.append(tmp)
        symbolic_shape = get_symbolic_shape(shape)
        symbolic_shapes.append(symbolic_shape)

    np_outs = np_lambda(*ops)
    if not isinstance(np_outs, (list, tuple)):
        np_outs = [np_outs]
    out_shapes = [out_.shape for out_ in np_outs]

    if len(ops) == 1:
        tvm_outs = with_tvm(
            lambda A: topi.einsum(
                subscripts, A, fcompute=fcompute, fcombine=fcombine, fidentity=fidentity
            ),
            symbolic_shapes,
            ops,
            out_shapes,
        )
    elif len(ops) == 2:
        tvm_outs = with_tvm(
            lambda A, B: topi.einsum(
                subscripts, A, B, fcompute=fcompute, fcombine=fcombine, fidentity=fidentity
            ),
            symbolic_shapes,
            ops,
            out_shapes,
        )
    elif len(ops) == 3:
        tvm_outs = with_tvm(
            lambda A, B, C: topi.einsum(
                subscripts, A, B, C, fcompute=fcompute, fcombine=fcombine, fidentity=fidentity
            ),
            symbolic_shapes,
            ops,
            out_shapes,
        )

    assert len(np_outs) == len(tvm_outs)
    for c1, c2 in zip(np_outs, tvm_outs):
        tvm.testing.assert_allclose(c1, c2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "equation,shapes,fcompute,fcombine,fidentity,np_lambda",
    [
        ("ij->ij, ij", [(5, 5)], None, None, None, lambda A: (A, A)),  # cannot have ii in output
        ("ij->ij, ij", [(5, 5)], None, None, None, lambda A: (A, A)),
        ("...j->...j, ...j", [(5, 5)], None, None, None, lambda A: (A, A)),
        (
            "ijk, jil->ijkl, ijkl",
            [(3, 4, 5), (4, 3, 2)],
            None,
            None,
            None,
            lambda A, B: (np.einsum("ijk, jil->ijkl", A, B), np.einsum("ijk, jil->ijkl", A, B)),
        ),
        ("ij, ij -> ij, ij", [(1, 4), (2, 4)], None, None, None, lambda A, B: (A * B, A * B)),
        (
            "...ij, ...jk -> ...ijk, ...ijk",
            [(1, 4), (4, 2)],
            None,
            None,
            None,
            lambda A, B: (
                np.einsum("...ij, ...jk->...ijk", A, B),
                np.einsum("...ij, ...jk->...ijk", A, B),
            ),
        ),
        (
            "...ij, ...ik -> ...ijk, ...ijk",
            [(1, 1, 1, 4), (1, 1, 1, 3)],
            None,
            None,
            None,
            lambda A, B: (
                np.einsum("...ij, ...ik -> ...ijk", A, B),
                np.einsum("...ij, ...ik -> ...ijk", A, B),
            ),
        ),
        (
            "...ik, ...jk, ...hk -> i...jhk, i...jhk",
            [(3, 4, 4), (1, 5, 3, 8, 4), (2, 5, 3, 6, 4)],
            None,
            None,
            None,
            lambda A, B, C: (
                np.einsum("...ik, ...jk, ...hk -> i...jhk", A, B, C),
                np.einsum("...ik, ...jk, ...hk -> i...jhk", A, B, C),
            ),
        ),
        (
            "ij,jk->ijk, ijk",
            [(2, 3), (3, 4)],
            None,
            None,
            None,
            lambda A, B: (np.einsum("ij,jk->ijk", A, B), np.einsum("ij,jk->ijk", A, B)),
        ),
        (
            "ij,jk,km->ijkm, ijkm",
            [(2, 3), (3, 4), (4, 5)],
            None,
            None,
            None,
            lambda A, B, C: (
                np.einsum("ij,jk,km->ijkm", A, B, C),
                np.einsum("ij,jk,km->ijkm", A, B, C),
            ),
        ),
    ],
)
def test_multi_outputs_without_reduction(
    equation, shapes, fcompute, fcombine, fidentity, np_lambda
):
    verify_einsum(equation, shapes, fcompute, fcombine, fidentity, np_lambda)


@pytest.mark.parametrize(
    "equation,shapes,fcompute,fcombine,fidentity,np_lambda",
    [
        (
            "ii -> ,",
            [(5, 5)],
            None,
            None,
            None,
            lambda A: (np.einsum("ii->", A), np.einsum("ii->", A)),
        ),
        (
            "ii->i, i",
            [(5, 5)],
            None,
            None,
            None,
            lambda A: (np.einsum("ii->i", A), np.einsum("ii->i", A)),
        ),
        (
            "ij->i, i",
            [(5, 5)],
            None,
            None,
            None,
            lambda A: (np.einsum("ij->i", A), np.einsum("ij->i", A)),
        ),
        (
            "...j->..., ...",
            [(5, 5)],
            None,
            None,
            None,
            lambda A: (np.einsum("...j->...", A), np.einsum("...j->...", A)),
        ),
        (
            "...j, j->..., ...",
            [(5, 5), (5,)],
            None,
            None,
            None,
            lambda A, B: (np.einsum("...j, j->...", A, B), np.einsum("...j, j->...", A, B)),
        ),
        (
            "..., ...-> ..., ...",
            [(), (2, 3)],
            None,
            None,
            None,
            lambda A, B: (np.einsum("..., ... -> ...", A, B), np.einsum("..., ... -> ...", A, B)),
        ),
        (
            "ijk, jil->kl, kl",
            [(3, 4, 5), (4, 3, 2)],
            None,
            None,
            None,
            lambda A, B: (np.einsum("ijk, jil->kl", A, B), np.einsum("ijk, jil->kl", A, B)),
        ),
        (
            "ij, ij -> i, i",
            [(1, 4), (2, 4)],
            None,
            None,
            None,
            lambda A, B: (np.einsum("ij, ij -> i", A, B), np.einsum("ij, ij -> i", A, B)),
        ),
        (
            "...ij, ...jk -> ...ik, ...ik",
            [(1, 4), (4, 2)],
            None,
            None,
            None,
            lambda A, B: (
                np.einsum("...ij, ...jk -> ...ik", A, B),
                np.einsum("...ij, ...jk -> ...ik", A, B),
            ),
        ),
        (
            "...ij, ...ik -> ...jk, ...jk",
            [(1, 1, 1, 4), (1, 1, 1, 3)],
            None,
            None,
            None,
            lambda A, B: (
                np.einsum("...ij, ...ik -> ...jk", A, B),
                np.einsum("...ij, ...ik -> ...jk", A, B),
            ),
        ),
        (
            "...ik, ...jk, ...hk -> i...jh, i...jh",
            [(3, 4, 4), (1, 5, 3, 8, 4), (2, 5, 3, 6, 4)],
            None,
            None,
            None,
            lambda A, B, C: (
                np.einsum("...ik, ...jk, ...hk -> i...jh", A, B, C),
                np.einsum("...ik, ...jk, ...hk -> i...jh", A, B, C),
            ),
        ),
        (
            "ij,jk->ik,ik",
            [(2, 3), (3, 4)],
            None,
            None,
            None,
            lambda A, B: (np.einsum("ij, jk->ik", A, B), np.einsum("ij, jk->ik", A, B)),
        ),
        (
            "ij,jk,km->im,im",
            [(2, 3), (3, 4), (4, 5)],
            None,
            None,
            None,
            lambda A, B, C: (
                np.einsum("ij,jk,km->im", A, B, C),
                np.einsum("ij,jk,km->im", A, B, C),
            ),
        ),
    ],
)
def test_multi_outpus_with_default_reduction(
    equation, shapes, fcompute, fcombine, fidentity, np_lambda
):
    verify_einsum(equation, shapes, fcompute, fcombine, fidentity, np_lambda)


@pytest.mark.parametrize(
    "equation,shapes,fcompute,fcombine,fidentity,np_lambda",
    [
        ("ij->ij", [(5, 5)], lambda x_ij: x_ij, None, None, lambda x: x),
        (
            "ij->ij",
            [(5, 5)],
            lambda x_ij: x_ij * x_ij + x_ij,
            None,
            None,
            lambda x: np.power(x, 2) + x,
        ),
        ("ij->ij", [(5, 5)], lambda x_ij: 1.0, None, None, lambda x: np.ones_like(x)),
        ("ij->i", [(5, 5)], lambda x_ij: x_ij * 2, None, None, lambda x: np.einsum("ij->i", 2 * x)),
        (
            "ij,jk->ik",
            [(2, 3), (3, 4)],
            lambda x_ij, y_jk: x_ij * y_jk + y_jk,
            None,
            None,
            lambda x, y: x @ y + np.einsum("jk->k", y),
        ),
        (
            "ij,jk,km->im",
            [(2, 3), (3, 4), (4, 5)],
            lambda x_ij, y_jk, z_km: x_ij * y_jk * z_km,
            None,
            None,
            lambda x, y, z: np.einsum("ij,jk,km->im", x, y, z),
        ),
        (
            ("ij->ij, ij"),
            [(5, 5)],
            lambda x_ij: (x_ij, x_ij * x_ij),
            None,
            None,
            lambda x: (x, np.power(x, 2)),
        ),
        (
            "ij->i, i",
            [(5, 5)],
            lambda x_ij: (x_ij, x_ij * x_ij),
            None,
            None,
            lambda x: (np.sum(x, axis=-1), np.sum(x * x, axis=-1)),
        ),
        (
            "ij,jk->ik, ik",
            [(2, 3), (3, 4)],
            lambda x_ij, y_jk: (x_ij * y_jk, x_ij + y_jk),
            None,
            None,
            lambda x, y: (
                x @ y,
                np.sum(x, axis=1, keepdims=True) + np.sum(y, axis=0, keepdims=True),
            ),
        ),
        (
            "ij,jk,km->im, im",
            [(2, 3), (3, 4), (4, 5)],
            lambda x_ij, y_jk, z_km: (x_ij * y_jk * z_km, x_ij * y_jk / z_km),
            None,
            None,
            lambda x, y, z: (
                np.einsum("ij,jk,km->im", x, y, z),
                np.einsum("ij,jk,km->im", x, y, 1 / z),
            ),
        ),
    ],
)
def test_customize_compute(equation, shapes, fcompute, fcombine, fidentity, np_lambda):
    verify_einsum(equation, shapes, fcompute, fcombine, fidentity, np_lambda)


@pytest.mark.parametrize(
    "equation,shapes,fcompute,fcombine,fidentity,np_lambda",
    [
        (
            "ij->ij",
            [(5, 5)],
            None,
            lambda x, y: x + y,  # no accumulate
            lambda dtype1: tvm.tir.const(0, dtype1),
            lambda x: x,
        ),
        (
            "ij->i",
            [(5, 5)],
            None,
            lambda x, y: x + y,
            lambda dtype1: tvm.tir.const(0, dtype1),
            lambda x: np.sum(x, axis=1),
        ),
        (
            "ij->i",
            [(5, 5)],
            None,
            lambda x, y: x * y,
            lambda dtype1: tvm.tir.const(1, dtype1),
            lambda x: np.prod(x, axis=1),
        ),
        (
            "ij->i",
            [(5, 5)],
            lambda x_ij: 2 * x_ij,
            lambda x, y: x * y,
            lambda dtype1: tvm.tir.const(1, dtype1),
            lambda x: np.prod(2 * x, axis=1),
        ),
        (
            "ij,jk->ik",
            [(2, 3), (3, 4)],
            lambda x_ij, y_jk: x_ij + y_jk,
            lambda x, y: x * y,
            lambda dtype1: tvm.tir.const(1, dtype1),
            lambda x, y: np.prod(np.expand_dims(x, -1) + np.expand_dims(y, 0), axis=1),
        ),
        (
            "ij,jk,km->im",
            [(2, 3), (3, 4), (4, 5)],
            None,
            lambda x, y: x + y,
            lambda dtype1: tvm.tir.const(0, dtype1),
            lambda x, y, z: np.einsum("ij,jk,km->im", x, y, z),
        ),
        (
            "ij->i, i",
            [(5, 5)],
            lambda x_ij: (x_ij, x_ij),
            lambda x, y: (x[0] + y[0], x[1] * y[1]),
            lambda dtype1, dtype2: (tvm.tir.const(0, dtype1), tvm.tir.const(1, dtype2)),
            lambda x: (np.sum(x, axis=-1), np.prod(x, axis=-1)),
        ),
    ],
)
def test_customize_combine(equation, shapes, fcompute, fcombine, fidentity, np_lambda):
    verify_einsum(equation, shapes, fcompute, fcombine, fidentity, np_lambda)


if __name__ == "__main__":
    tvm.testing.main()
