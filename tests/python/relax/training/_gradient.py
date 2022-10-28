from __future__ import annotations

from tvm.relay.op import register_gradient
import tvm.relax.op.nn as nn
from tvm.relax.op import (
	collapse_sum_like,
	log,
	multiply,
	negative,
	sub,
	transpose,
	ones_like
)

from tvm.relax.op.nn import (
	gradrelu_,
	softmax
)

@register_gradient("relax.add")
def add_grad(orig, grad):
	"""Returns [grad, grad]"""
	return [collapse_sum_like(grad, orig.args[0]), collapse_sum_like(grad, orig.args[1])]

@register_gradient("relax.sub")
def sub_grad(orig, grad):
	"""Returns [grad, -grad]"""
	return [collapse_sum_like(grad, orig.args[0]), collapse_sum_like(negative(grad), orig.args[1])]

@register_gradient("relax.multiply")
def multiply_grad(orig, grad):
	"""Returns [grad * y, grad * x]"""
	x, y = orig.args
	return [collapse_sum_like(multiply(grad, y), x), collapse_sum_like(multiply(grad, x), y)]

@register_gradient("relax.transpose")
def transpose_grad(orig, grad):
	"""Returns grad transposed over the complement of original transpose axes"""
	"""TODO: Do not support more than one dimensions"""
	return [transpose(grad)]

@register_gradient("relax.nn.relu")
def relu_grad(orig, grad):
	"""Returns grad * (select(x < 0, 0, 1))."""
	return [multiply(grad, gradrelu_(orig.args[0]))]


@register_gradient("relax.nn.matmul")
def matmul_grad(orig, grad):
	"""Returns [grad' @ tensor_b, tensor_a @ grad']"""
	tensor_a, tensor_b = orig.args
	return [
		collapse_sum_like(nn.matmul(grad, transpose(tensor_b)), tensor_a),
		collapse_sum_like(nn.matmul(transpose(tensor_a), grad), tensor_b),
	]

@register_gradient("relax.sum")
def sum_grad(orig, grad):
	"""Returns [grad * ones_like(x)]"""
	return [multiply(grad, ones_like(orig.args[0]))]


# @register_gradient("relax.nn.softmax")
# def softmax_grad(orig, grad):
# 	"""Gradient of softmax"""
# 	return [(grad - _sum(grad * orig, orig.attrs.axis, True)) * orig]


# @register_gradient("relax.nn.cross_entropy")
# def cross_entropy_grad(orig, grad):
# 	x, y = orig.args
# 	shape = shape_of(x)
# 	batch_size = take(shape, const(0, dtype="int32"), axis=0)
# 	grad = grad / batch_size.astype(x.checked_type.dtype)
# 	return [-grad * y / x, -grad * log(x)]

# softmax_cross_entropy(z, y)
@register_gradient("relax.nn.softmax_cross_entropy")
def softmax_cross_entropy_grad(orig, grad):
	y_hat = softmax(orig.args[0])
	return [multiply(grad, sub(y_hat, orig.args[1])), multiply(grad, negative(log(y_hat)))]
	# return [sub(y_hat, orig.args[1]), negative(log(y_hat))]
