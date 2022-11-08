from __future__ import annotations

import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm import topi, relax, te
from tvm.script import tir as T
from tvm.script import relax as R

import _gradient


@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: IRModule, op_map) -> None:
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }


    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        if isinstance(call, relax.Call) and call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()


def map_dense(bb, call):
    x, w = call.args
    return bb.call_te(topi.nn.dense, x, w)

def map_add(bb, call):
    a, b = call.args
    return bb.call_te(topi.add, a, b)

def map_subtract(bb, call):
    return bb.call_te(topi.subtract, call.args[0], call.args[1])

def map_multiply(bb, call):
    return bb.call_te(topi.multiply, call.args[0], call.args[1])

def map_transpose(bb, call):
    return bb.call_te(topi.transpose, call.args[0])

def map_sum(bb, call):
    return bb.call_te(topi.sum, call.args[0])

def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])

def map_gradrelu_(bb, call):
    def _gradrelu_(x):
        return te.compute(shape=x.shape, fcompute=lambda *indices: te.if_then_else(x(*indices)>0, 1.0, 0.0), name="gradrelu_")
    return bb.call_te(_gradrelu_, call.args[0])

def map_matmul(bb, call):
    return bb.call_te(topi.matmul, call.args[0], call.args[1])

def map_softmax(bb, call):
    return bb.call_te(topi.nn.softmax, call.args[0])

def te_cross_entropy(x, y):
    # i = te.reduce_axis((0, 10), name="i")
    # result = te.compute(shape=(), fcompute=lambda : te.sum(-y[0, i] * te.log(x[0, i]), axis=i), name="cross_entropy")
    # return te.compute(shape=result.shape, fcompute=lambda *indices: te.if_then_else(te.isnan(result(*indices)), 0.0, result(*indices)), name="crossent_process")
    return -topi.sum(topi.log(x) * y)

def map_cross_entropy(bb, call):
    return bb.call_te(te_cross_entropy, call.args[0], call.args[1])

def map_softmax_cross_entropy(bb, call):
    func = lambda x, y: te_cross_entropy(topi.nn.softmax(x), y)
    return bb.call_te(func, call.args[0], call.args[1])

def map_sigmoid(bb, call):
    return bb.call_te(topi.sigmoid, call.args[0])

def map_tanh(bb, call):
    return bb.call_te(topi.tanh, call.args[0])

def map_negative(bb, call):
    return bb.call_te(topi.negative, call.args[0])

def map_log(bb, call):
    return bb.call_te(topi.log, call.args[0])

def map_ones_like(bb, call):
    def te_ones_like(x):
        return topi.full_like(x, 1.0)
    return bb.call_te(te_ones_like, call.args[0])

def map_zeros_like(bb, call):
    def te_zeros_like(x):
        return topi.full_like(x, 0.0)
    return bb.call_te(te_zeros_like, call.args[0])

def map_collapse_sum_like(bb, call):
    def te_collapse_sum_like(x, y):
        return topi.collapse_sum(x, y.shape)
    return bb.call_te(te_collapse_sum_like, call.args[0], call.args[1])

def map_zeros(bb, call):
    shape_values = [prim_expr.value for prim_expr in call.args[0].values]
    return tvm.relay.const(np.zeros(shape_values))

def map_ones(bb, call):
    shape_values = [prim_expr.value for prim_expr in call.args[0].values]
    return tvm.relay.const(np.ones(shape_values))


op_map = {
  "relax.nn.dense": map_dense,
  "relax.add": map_add,
  "relax.subtract": map_subtract,
  "relax.multiply": map_multiply,
  "relax.transpose": map_transpose,
  "relax.nn.relu": map_relu,
  "relax.nn.gradrelu_": map_gradrelu_,
  "relax.nn.matmul": map_matmul,
  "relax.nn.softmax": map_softmax,
  "relax.nn.cross_entropy": map_cross_entropy,
  "relax.nn.softmax_cross_entropy": map_softmax_cross_entropy,
  "relax.nn.sigmoid": map_sigmoid,
  "relax.nn.tanh": map_tanh,
  "relax.negative": map_negative,
  "relax.ones_like": map_ones_like,
  "relax.zeros_like": map_zeros_like,
  "relax.collapse_sum_like": map_collapse_sum_like,
  "relax.log": map_log,
  "relax.sum": map_sum,
  "relax.zeros": map_zeros,
  "relax.ones": map_ones
}

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()
