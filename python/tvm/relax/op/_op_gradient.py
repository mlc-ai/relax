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
"""Gradient definitions for Relax operators"""
from typing import List
from tvm import relax
from tvm.relax.expr import Call, Var, Expr
from ...tir import PrimExpr

from .base import register_gradient

from .unary import (
    log,
    negative,
)
from .binary import (
    subtract,
    multiply,
    divide,
    less,
)
from .statistical import sum as _sum
from .statistical import mean
from .create import (
    zeros,
    ones,
)
from .search import where
from .linear_algebra import matmul
from .manipulate import (
    collapse_sum_to,
    broadcast_to,
    permute_dims,
    expand_dims,
    concat,
    split,
)
from .nn import (
    softmax,
)


# Arith
@register_gradient("relax.add")
def add_grad(orig: Call, grad: Var):
    """Gradient of add.

    Forward Form:
        z = relax.add(x, y)

    Backward:
        Returns [z_grad, z_grad].
    """
    return [
        collapse_sum_to(grad, orig.args[0].struct_info.shape),
        collapse_sum_to(grad, orig.args[1].struct_info.shape),
    ]


@register_gradient("relax.subtract")
def subtract_grad(orig: Call, grad: Var):
    """Gradient of subtract.

    Forward Form:
        z = relax.subtract(x, y)

    Backward:
        Returns [z_grad, -z_grad].
    """
    return [
        collapse_sum_to(grad, orig.args[0].struct_info.shape),
        collapse_sum_to(negative(grad), orig.args[1].struct_info.shape),
    ]


@register_gradient("relax.multiply")
def multiply_grad(orig: Call, grad: Var):
    """Gradient of multiply.

    Forward Form:
        z = relax.multiply(x, y)

    Backward:
        Returns [z_grad * y, z_grad * x].
    """
    x, y = orig.args
    return [
        collapse_sum_to(multiply(grad, y), x.struct_info.shape),
        collapse_sum_to(multiply(grad, x), y.struct_info.shape),
    ]


@register_gradient("relax.sigmoid")
def sigmoid_grad(orig: Call, grad: Var):
    """Gradient of sigmoid.

    Forward Form:
        y = relax.sigmoid(x)

    Backward:
        Returns [y_grad * y * (1 - y)].
    """
    return [
        multiply(
            grad,
            multiply(
                orig,
                subtract(
                    ones(orig.args[0].struct_info.shape, orig.args[0].struct_info.dtype), orig
                ),
            ),
        )
    ]


@register_gradient("relax.tanh")
def tanh_grad(orig: Call, grad: Var):
    """Gradient of tanh.

    Forward Form:
        y = relax.tanh(x)

    Backward:
        Returns [y_grad * (1 - y * y)].
    """
    return [
        multiply(
            grad,
            subtract(
                ones(orig.args[0].struct_info.shape, orig.args[0].struct_info.dtype),
                multiply(orig, orig),
            ),
        )
    ]


# Statistical
@register_gradient("relax.sum")
def sum_grad(orig: Call, grad: Var):
    """Gradient of sum.

    Forward Form:
        y = relax.sum(x, axis, keepdims)

    Backward:
        Returns [broadcast_to(y_grad, x.shape)].
        If `keepdims=False`, the summed axis will be added back.
    """
    axis = orig.attrs["axis"]
    keepdims = orig.attrs["keepdims"]
    if not keepdims and axis:
        grad = expand_dims(grad, [int(ax) for ax in axis])
    return [broadcast_to(grad, orig.args[0].struct_info.shape)]


# Manipulate
@register_gradient("relax.permute_dims")
def permute_dims_grad(orig: Call, grad: Var):
    """Gradient of permute_dims.

    Forward Form:
        y = relax.permute_dims(x, axes)

    Backward:
        Returns grad transposed over the **inverse permutation** of the original permute_dims axes.
    """
    axes = orig.attrs["axes"]
    if axes:
        dims = len(axes)
        new_axes = [0] * dims
        for i in range(dims):
            new_axes[int(axes[i])] = i
        return [permute_dims(grad, axes=new_axes)]
    else:
        return [permute_dims(grad)]

# TODO(yixin, chaofan): handle symbolic shape
@register_gradient("relax.concat")
def concat_grad(orig: Call, grad: Var):
    """Gradient of concat.

    Forward Form:
        y = concat((x1, x2, x3), axis)

    Backward:
        Returns [split(y_grad, [x1.shape[axis], x1.shape[axis] + x2.shape[axis]], axis)].
    """
    axis = orig.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    split_indices: List[PrimExpr] = []
    for i in range(len(orig.args[0]) - 1):
        sinfo = orig.args[0].struct_info.fields[i]
        index = sinfo.shape[axis]
        if i > 0:
            index += split_indices[i - 1]
        split_indices.append(index)
    return [split(grad, split_indices, axis)]


@register_gradient("relax.split")
def split_grad(orig: Call, grad: Var):
    """Gradient of split.

    Forward Form:
        y = split(x, indices, axis)

    Backward:
        Returns [concat(y_grad, axis)].
    """
    axis = orig.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    return [concat(grad, axis)]


# Linear Algebra
@register_gradient("relax.matmul")
def matmul_grad(orig: Call, grad: Var):
    """Gradient of matmul.

    Forward Form:
        c = relax.matmul(a, b)

    Backward:
        Generally, returns [c_grad @ b^T, a^T @ c_grad].
        Here we only transpose the last two dimensions because of the definition
        of batch matmul. Note that ndim=1 should be treaded specially.
    """

    tensor_a, tensor_b = orig.args

    a_dim = len(tensor_a.struct_info.shape)
    b_dim = len(tensor_b.struct_info.shape)

    def _transpose_last_two_dim(tensor, ndim):
        """Helper function for reversing the last two dimensions."""
        assert ndim > 1
        return permute_dims(
            tensor, axes=[i if i < ndim - 2 else 2 * ndim - 3 - i for i in range(ndim)]
        )

    if a_dim > 1 and b_dim > 1:
        a_grad = matmul(grad, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(_transpose_last_two_dim(tensor_a, a_dim), grad)
    elif a_dim == 1 and b_dim > 1:
        a_expand = expand_dims(tensor_a, 1)
        grad_expand = expand_dims(grad, -2)
        a_grad = matmul(grad_expand, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(a_expand, grad_expand)
    elif b_dim == 1 and a_dim > 1:
        b_expand = expand_dims(tensor_b, 0)
        grad_expand = expand_dims(grad, -1)
        a_grad = matmul(grad_expand, b_expand)
        b_grad = mean(
            matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        )  # squeeze last dim
        # legalizer now not support squeeze
        # squeeze(
        #     matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        # )
    else:
        assert a_dim == 1 and b_dim == 1
        a_grad = multiply(grad, tensor_b)
        b_grad = multiply(grad, tensor_a)

    return [
        collapse_sum_to(a_grad, tensor_a.struct_info.shape),
        collapse_sum_to(b_grad, tensor_b.struct_info.shape),
    ]


# NN
@register_gradient("relax.nn.relu")
def relu_grad(orig: Call, grad: Var):
    """Gradient of relu.

    Forward Form:
        y = relax.relu(x)

    Backward:
        Returns [y_grad * (where(x < 0, 0, 1))].
    """
    x = orig.args[0]
    x_zeros = zeros(x.struct_info.shape, x.struct_info.dtype)
    x_ones = ones(x.struct_info.shape, x.struct_info.dtype)
    return [where(less(x, x_zeros), x_zeros, multiply(x_ones, grad))]


@register_gradient("relax.nn.softmax")
def softmax_grad(orig: Call, grad: Var):
    """Gradient of softmax.

    Forward Form:
        y = relax.softmax(x, axis)

    Backward:
        Returns [(y_grad - sum(y_grad * y, axis, keepdims=True)) * y]
    """
    return [multiply(subtract(grad, _sum(multiply(grad, orig), orig.attrs.axis, True)), orig)]


@register_gradient("relax.nn.log_softmax")
def log_softmax_grad(orig: Call, grad: Var):
    """Gradient of log_softmax.

    Forward Form:
        y = relax.log_softmax(x, axis)

    Backward:
        Returns [y_grad - sum(y_grad, axis, keepdims=True) * softmax(x)]
    """
    x_softmax = softmax(orig.args[0], orig.attrs.axis)
    return [subtract(grad, multiply(_sum(grad, orig.attrs.axis, True), x_softmax))]


def _divide_batch(x: Expr, expr: Expr):
    if x.struct_info.ndim > 1:
        # TODO(chaofan, yixin): support symbolic shape
        batch_size = int(x.struct_info.shape[0])
        # batch_size = take(shape_of(x), relax.const(0, dtype="int32"), axis=0)
        # expr = divide(expr, batch_size)
        expr = divide(expr, relax.const(batch_size, dtype=expr.struct_info.dtype))
    return expr


@register_gradient("relax.nn.cross_entropy_without_logits")
def cross_entropy_without_logits_grad(orig: Call, grad: Var):
    """Gradient of cross_entropy_without_logits.

    Forward Form:
        z = cross_entropy_without_logits(x, y)

    Backward:
        Returns [-z_grad * y / x, -z_grad * log(x)].
        If it has batch_size N, the results should divide by N.
    """
    x, y = orig.args
    grad = _divide_batch(x, grad)
    return [negative(multiply(grad, divide(y, x))), negative(multiply(grad, log(x)))]


@register_gradient("relax.nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(orig: Call, grad: Var):
    """Gradient of cross_entropy_without_logits.

    Forward Form:
        z = cross_entropy_with_logits(x, y)

    Backward:
        Returns [-z_grad * y, -z_grad * x].
        If it has batch_size N, the results should divide by N.
    """
    x, y = orig.args
    grad = _divide_batch(x, grad)
    return [negative(multiply(grad, y)), negative(multiply(grad, x))]
