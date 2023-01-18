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
# pylint: disable=unused-argument, redefined-builtin
"""Gradient definitions for Relax operators"""
from typing import List
from tvm import relax
from tvm.arith import Analyzer
from tvm.relax.expr import Call, Var, Expr, ShapeExpr
from tvm._ffi.base import TVMError

from ..block_builder import BlockBuilder
from ...tir import PrimExpr
from .base import register_gradient

from .unary import (
    log,
    negative,
    exp,
)
from .binary import (
    subtract,
    multiply,
    divide,
    less,
)
from .statistical import sum
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
    squeeze,
)


def _get_shape(expr: Expr) -> ShapeExpr:
    """Get the shape from a Tensor expr."""
    try:
        shape = expr.struct_info.shape
    except Exception as error:
        raise TVMError(
            f"Get the shape of {expr} failed. Please normalize it first and ensure it is a Tensor."
        ) from error
    return shape


def _fit_shape(expr: Expr, expr_shape: ShapeExpr, target: Expr) -> Expr:
    target_shape = _get_shape(target)
    expr_sinfo = expr_shape.struct_info
    target_sinfo = target_shape.struct_info
    assert isinstance(expr_sinfo, relax.ShapeStructInfo)
    assert isinstance(target_sinfo, relax.ShapeStructInfo)

    def _check_shape_equal():
        if len(expr_sinfo.values) != len(target_sinfo.values):
            return False
        analyzer = Analyzer()
        for i, field in enumerate(expr_sinfo.values):
            if not analyzer.can_prove_equal(field, target_sinfo.values[i]):
                return False
        return True

    return expr if _check_shape_equal() else collapse_sum_to(expr, target_shape)


# Arith
@register_gradient("relax.add")
def add_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of add.

    Forward Form:
        z = relax.add(x, y)

    Backward:
        Returns [z_output_grad, z_grad].
    """
    output_grad_shape = _get_shape(output_grad)
    return [
        _fit_shape(output_grad, output_grad_shape, orig_call.args[0]),
        _fit_shape(output_grad, output_grad_shape, orig_call.args[1]),
    ]


@register_gradient("relax.subtract")
def subtract_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of subtract.

    Forward Form:
        z = relax.subtract(x, y)

    Backward:
        Returns [z_output_grad, -z_grad].
    """
    output_grad_shape = _get_shape(output_grad)
    return [
        _fit_shape(output_grad, output_grad_shape, orig_call.args[0]),
        _fit_shape(negative(output_grad), output_grad_shape, orig_call.args[1]),
    ]


@register_gradient("relax.multiply")
def multiply_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of multiply.

    Forward Form:
        z = relax.multiply(x, y)

    Backward:
        Returns [z_grad * y, z_grad * x].
    """
    x, y = orig_call.args
    output_grad_shape = _get_shape(output_grad)
    return [
        _fit_shape(multiply(output_grad, y), output_grad_shape, x),
        _fit_shape(multiply(output_grad, x), output_grad_shape, y),
    ]


@register_gradient("relax.sigmoid")
def sigmoid_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of sigmoid.

    Forward Form:
        y = relax.sigmoid(x)

    Backward:
        Returns [y_grad * y * (1 - y)].
    """
    x_ones = ones(_get_shape(orig_call.args[0]), orig_call.args[0].struct_info.dtype)
    return [
        multiply(
            output_grad,
            multiply(orig_var, subtract(x_ones, orig_var)),
        )
    ]


@register_gradient("relax.tanh")
def tanh_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of tanh.

    Forward Form:
        y = relax.tanh(x)

    Backward:
        Returns [y_grad * (1 - y * y)].
    """
    x_ones = ones(_get_shape(orig_call.args[0]), orig_call.args[0].struct_info.dtype)
    return [
        multiply(
            output_grad,
            subtract(x_ones, multiply(orig_var, orig_var)),
        )
    ]


# Statistical
@register_gradient("relax.sum")
def sum_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of sum.

    Forward Form:
        y = relax.sum(x, axis, keepdims)

    Backward:
        Returns [broadcast_to(y_output_grad, x.shape)].
        If `keepdims=False`, the summed axis will be added back.
    """
    axis = orig_call.attrs["axis"]
    keepdims = orig_call.attrs["keepdims"]
    if not keepdims and axis:
        output_grad = expand_dims(output_grad, axis)
    return [broadcast_to(output_grad, _get_shape(orig_call.args[0]))]


# Manipulate
@register_gradient("relax.permute_dims")
def permute_dims_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of permute_dims.

    Forward Form:
        y = relax.permute_dims(x, axes)

    Backward:
        Returns grad transposed over the **inverse permutation** of the original permute_dims axes.
    """
    axes = orig_call.attrs["axes"]
    if axes:
        dims = len(axes)
        new_axes = [0] * dims
        for i in range(dims):
            new_axes[int(axes[i])] = i
        return [permute_dims(output_grad, axes=new_axes)]
    else:
        return [permute_dims(output_grad)]


# TODO(yixin, chaofan): handle symbolic shape
@register_gradient("relax.concat")
def concat_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of concat.

    Forward Form:
        y = concat((x1, x2, x3), axis)

    Backward:
        Returns [split(y_output_grad, [x1.shape[axis], x1.shape[axis] + x2.shape[axis]], axis)].
    """
    axis = orig_call.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    split_indices: List[PrimExpr] = []
    sinfo = orig_call.args[0].struct_info
    assert isinstance(sinfo, relax.TupleStructInfo)
    for i in range(len(sinfo.fields) - 1):
        tensor_sinfo = sinfo.fields[i]
        assert isinstance(tensor_sinfo, relax.TensorStructInfo)
        assert tensor_sinfo.shape is not None
        index = tensor_sinfo.shape[axis]
        if i > 0:
            index += split_indices[i - 1]
        split_indices.append(index)
    return [split(output_grad, split_indices, axis)]


@register_gradient("relax.split")
def split_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of split.

    Forward Form:
        y = split(x, indices, axis)

    Backward:
        Returns [concat(y_output_grad, axis)].
    """
    axis = orig_call.attrs["axis"]
    assert axis is not None
    axis = int(axis)
    return [concat(output_grad, axis)]


# Linear Algebra
@register_gradient("relax.matmul")
def matmul_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of matmul.

    Forward Form:
        c = relax.matmul(a, b)

    Backward:
        Generally, returns [c_grad @ b^T, a^T @ c_grad].
        Here we only transpose the last two dimensions because of the definition
        of batch matmul. Note that ndim=1 should be treaded specially.
    """

    tensor_a, tensor_b = orig_call.args

    a_dim = len(_get_shape(tensor_a))
    b_dim = len(_get_shape(tensor_b))

    def _transpose_last_two_dim(tensor, ndim):
        """Helper function for reversing the last two dimensions."""
        assert ndim > 1
        return permute_dims(
            tensor, axes=[i if i < ndim - 2 else 2 * ndim - 3 - i for i in range(ndim)]
        )

    if a_dim > 1 and b_dim > 1:
        a_grad = matmul(output_grad, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(_transpose_last_two_dim(tensor_a, a_dim), output_grad)
    elif a_dim == 1 and b_dim > 1:
        a_expand = expand_dims(tensor_a, 1)
        grad_expand = expand_dims(output_grad, -2)
        a_grad = matmul(grad_expand, _transpose_last_two_dim(tensor_b, b_dim))
        b_grad = matmul(a_expand, grad_expand)
    elif b_dim == 1 and a_dim > 1:
        b_expand = expand_dims(tensor_b, 0)
        grad_expand = expand_dims(output_grad, -1)
        a_grad = matmul(grad_expand, b_expand)
        b_grad = squeeze(
            matmul(_transpose_last_two_dim(tensor_a, a_dim), grad_expand), axis=-1
        )  # squeeze last dim
    else:
        assert a_dim == 1 and b_dim == 1
        a_grad = multiply(output_grad, tensor_b)
        b_grad = multiply(output_grad, tensor_a)

    output_grad_shape = _get_shape(output_grad)

    return [
        _fit_shape(a_grad, output_grad_shape, tensor_a),
        _fit_shape(b_grad, output_grad_shape, tensor_b),
    ]


# NN
@register_gradient("relax.nn.relu")
def relu_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of relu.

    Forward Form:
        y = relax.relu(x)

    Backward:
        Returns [y_grad * (where(x < 0, 0, 1))].
    """
    x = orig_call.args[0]
    x_shape = _get_shape(x)
    x_zeros = zeros(x_shape, x.struct_info.dtype)
    x_ones = ones(x_shape, x.struct_info.dtype)
    return [where(less(x, x_zeros), x_zeros, multiply(x_ones, output_grad))]


@register_gradient("relax.nn.softmax")
def softmax_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of softmax.

    Forward Form:
        y = relax.softmax(x, axis)

    Backward:
        Returns [(y_grad - sum(y_grad * y, axis, keepdims=True)) * y]
    """
    return [
        multiply(
            subtract(output_grad, sum(multiply(output_grad, orig_var), orig_call.attrs.axis, True)),
            orig_var,
        )
    ]


@register_gradient("relax.nn.log_softmax")
def log_softmax_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of log_softmax.

    Forward Form:
        y = relax.log_softmax(x, axis)

    Backward:
        Returns [y_grad - sum(y_output_grad, axis, keepdims=True) * softmax(x)]
    """
    x_softmax = exp(orig_var)
    return [
        subtract(output_grad, multiply(sum(output_grad, orig_call.attrs.axis, True), x_softmax))
    ]


def _divide_batch(x: Expr, expr: Expr):
    if x.struct_info.ndim > 1:
        # TODO(chaofan, yixin): support symbolic shape
        x_shape = _get_shape(x)
        batch_size = int(x_shape[0])
        # batch_size = take(shape_of(x), relax.const(0, dtype="int32"), axis=0)
        # expr = divide(expr, batch_size)
        expr = divide(expr, relax.const(batch_size, dtype=expr.struct_info.dtype))
    return expr


@register_gradient("relax.nn.cross_entropy_without_logits")
def cross_entropy_without_logits_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of cross_entropy_without_logits.

    Forward Form:
        z = cross_entropy_without_logits(x, y)

    Backward:
        Returns [-z_grad * y / x, -z_grad * log(x)].
        If it has batch_size N, the results should divide by N.
    """
    x, y = orig_call.args
    output_grad = _divide_batch(x, output_grad)
    return [negative(multiply(output_grad, divide(y, x))), negative(multiply(output_grad, log(x)))]


@register_gradient("relax.nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(
    orig_var: Var,
    orig_call: Call,
    output_grad: Var,
    ctx: BlockBuilder,
):
    """Gradient of cross_entropy_without_logits.

    Forward Form:
        z = cross_entropy_with_logits(x, y)

    Backward:
        Returns [-z_grad * y, -z_grad * x].
        If it has batch_size N, the results should divide by N.
    """
    x, y = orig_call.args
    output_grad = _divide_batch(x, output_grad)
    return [negative(multiply(output_grad, y)), negative(multiply(output_grad, x))]
