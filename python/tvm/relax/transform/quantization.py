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
# pylint: disable=invalid-name
"""Relax quantization passes."""
import tvm
from tvm import topi, relax
from tvm.ir.module import IRModule
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.analysis import remove_all_unused
from tvm.ir import Array
from tvm.relax import Tuple
from tvm.tir import IntImm


def te_encode_i4(x, group_size=64):
    assert len(x.shape) == 2
    assert x.shape[1] % group_size == 0
    x_reshape = topi.reshape(x, (x.shape[0], x.shape[1] // group_size, group_size))
    x_max = topi.max(x_reshape, axis=2, keepdims=True)
    x_min = topi.min(x_reshape, axis=2, keepdims=True)
    dtype_bits = 4
    x_scaled = topi.round((2**dtype_bits - 1) * (x_reshape - x_min) / (x_max - x_min))
    x_int = topi.cast(x_scaled, "uint8")
    x_first_half = topi.strided_slice(x_int, (0,), (x_int.shape[2] // 2,), (1,), axes=(2,))
    x_second_half = topi.strided_slice(
        x_int, (x_int.shape[2] // 2,), (x_int.shape[2],), (1,), axes=(2,)
    )
    x_first_half = topi.left_shift(x_first_half, IntImm(dtype="uint8", value=dtype_bits))
    x_masked = topi.bitwise_or(x_first_half, x_second_half)

    return (x_masked, x_max, x_min)


def te_decode_i4(x, x_max, x_min):
    assert len(x.shape) == 3
    assert len(x_max.shape) == 3
    assert len(x_min.shape) == 3
    assert x.shape[0] == x_max.shape[0] and x_min.shape[0] == x_max.shape[0]
    assert x.shape[1] == x_max.shape[1] and x_min.shape[1] == x_max.shape[1]

    dtype_bits = 4
    x_first_half = topi.right_shift(x, IntImm(dtype="uint8", value=dtype_bits))
    x_second_half = topi.bitwise_and(x, IntImm(dtype="uint8", value=2**dtype_bits - 1))
    x_concat = topi.concatenate([x_first_half, x_second_half], axis=2)
    x_float = topi.cast(x_concat, "float32")
    x_rescaled = x_float * (x_max - x_min) / (2**dtype_bits - 1) + x_min
    x_reshaped = topi.reshape(
        x_rescaled, (x_rescaled.shape[0], x_rescaled.shape[1] * x_rescaled.shape[2])
    )
    return x_reshaped


@tvm.transform.module_pass(opt_level=3, name="GroupQuantize")
class GroupQuantize:
    def __init__(self, group_size: int = 64) -> None:
        self.group_size = group_size

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        @mutator
        class QuantizeMutator(PyExprMutator):
            def __init__(self, mod: IRModule, group_size: int):
                super().__init__(mod)
                self.mod = mod
                self._params = set()
                self.group_size = group_size

            def emit_te_encode_decode(self, x):
                encoded_data = self.builder_.emit_te(
                    te_encode_i4, x, group_size=self.group_size, primfunc_name_hint="encode"
                )
                data = self.builder_.emit(relax.TupleGetItem(encoded_data, 0))
                data_max = self.builder_.emit(relax.TupleGetItem(encoded_data, 1))
                data_min = self.builder_.emit(relax.TupleGetItem(encoded_data, 2))
                decoded_data = self.builder_.emit_te(
                    te_decode_i4,
                    data,
                    data_max,
                    data_min,
                    primfunc_name_hint="decode",
                    primfunc_attrs={"StopLifting": True},
                )
                return decoded_data

            def transform(self) -> IRModule:
                for global_var, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue
                    if not "num_input" in func.attrs:
                        continue
                    num_inputs = func.attrs["num_input"]
                    for i in range(int(num_inputs), len(func.params)):
                        self._params.add(func.params[i])
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(global_var, updated_func)
                return self.builder_.get()

            def process_args(self, args):
                if isinstance(args, (Array, Tuple)):
                    updated = False
                    new_args = []
                    for arg in args:
                        new_arg, arg_updated = self.process_args(arg)
                        new_args.append(new_arg)
                        updated = updated or arg_updated
                    return (Tuple(new_args) if isinstance(args, Tuple) else new_args), updated
                elif isinstance(args, relax.Var):
                    if args in self._params:
                        return self.emit_te_encode_decode(args), True
                    else:
                        return args, False
                else:
                    return args, False

            def visit_call_(self, call):
                call = self.visit_expr_post_order(call)
                new_args, updated = self.process_args(call.args)
                if not updated:
                    return call
                new_call = relax.Call(call.op, new_args, call.attrs, call.sinfo_args, call.span)
                return new_call

        return QuantizeMutator(mod, self.group_size).transform()
