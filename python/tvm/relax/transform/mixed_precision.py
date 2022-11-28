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
# pylint: disable=line-too-long,unused-argument
"""Default behavior for ops in mixed_precision pass. Import this file to use."""
import copy
from typing import List

from tvm.relay import Call
from tvm.relax.op import dense, matmul, conv2d
from tvm.relay.op import register_mixed_precision_conversion

# MIXED_PRECISION_ALWAYS ops should always be done in lower precision due to the speed and memory
# savings. MIXED_PRECISION_FOLLOW ops can be done in lower precision but don't have speedups to
# justify a cast. MIXED_PRECISION_NEVER colored ops should not be done in lower precision due to
# numerical reasons.
MIXED_PRECISION_ALWAYS = 0
MIXED_PRECISION_FOLLOW = 1
MIXED_PRECISION_NEVER = 2

# Default lists inspired from TF's classifications:
# github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/core/grappler/optimizers/auto_mixed_precision_lists.h
# They have a bias toward Nvidia Tensor Cores so modify lists per your hardware choice.
DEFAULT_ALWAYS_LIST = ["relax.nn.dense", "relax.nn.conv2d", "relax.nn.matmul"]
DEFAULT_FOLLOW_LIST = [
    "relax.nn.flatten",
    "relax.nn.batch_norm",
    "relax.nn.dropout",
    "relax.nn.max_pool2d",
    "relax.ewise_fma",
    "relax.transpose",
    "relax.reshape",
    "relax.expand_dims",
    "relax.squeeze",
    "relax.unique",
    "relax.nn.relu",
    "relax.nn.gelu",
    "relax.multiply",
    "relax.add",
    "relax.nn.silu",
    "relax.sqrt",
    "relax.divide",
    "relax.subtract",
    "relax.strided_slice",
    "relax.sin",
    "relax.cos",
    "relax.concatenate",
    "relax.image.resize2d",
    "relax.cast",
    "relax.broadcast_to",
]
DEFAULT_NEVER_LIST = ["relax.nn.softmax", "relax.nn.layer_norm", "relax.sum", "relax.mean"]

# Returns a decorator which registers for every given op, the function under FTVMMixedPrecisionConversionType


def register_func_to_op_list(list_ops: List):
    def decorator(func):
        for op_name in list_ops:
            register_mixed_precision_conversion(op_name, func=func)

    return decorator


def get_generic_out_dtypes(call_node: "relay.Call", expected_out_dtype: str) -> List:
    """A function which returns output dtypes in a way which works for most ops.

    Parameters
    ---------
    call_node: relay.Call
        The call node containing the op.

    expected_out_dtype: str
        The output dtype to use.

    Returns
    -------
    output_dtypes : [str]
        A list of output dtype.
    """
    if hasattr(call_node.attrs, "out_dtype"):
        if call_node.op.name == "relax.nn.dense":
            adjust_call = dense(
                call_node.args[0],
                call_node.args[1],
                units=call_node.attrs.units,
                out_dtype=expected_out_dtype,
            )
        elif call_node.op.name == "relax.nn.conv2d":
            adjust_call = conv2d(
                call_node.args[0],
                call_node.args[1],
                channels=call_node.attrs.channels,
                kernel_size=call_node.attrs.kernel_size,
                strides=call_node.attrs.strides,
                padding=call_node.attrs.padding,
                dilation=call_node.attrs.dilation,
                groups=call_node.attrs.groups,
                data_layout=call_node.attrs.data_layout,
                kernel_layout=call_node.attrs.kernel_layout,
                out_layout=call_node.attrs.out_layout,
                out_dtype=expected_out_dtype,
            )
        elif call_node.op.name == "relax.nn.matmul":
            adjust_call = matmul(
                call_node.args[0],
                call_node.args[1],
                out_dtype=expected_out_dtype,
            )
        else:
            raise ValueError("Unsupported op for get_generic_out_dtypes", call_node.op.name)
        return [
            True,
            adjust_call,
        ]
    else:
        return [False, call_node]


# Functions for FTVMMixedPrecisionConversionType which
# Take in CallNodes and a DType and returns a conversion type,
# an accumulation dtype, and an output_dtype.
@register_func_to_op_list(list_ops=DEFAULT_ALWAYS_LIST)
def generic_always_op(call_node: "relay.Call", expected_out_dtype: str) -> List:
    return [MIXED_PRECISION_ALWAYS] + get_generic_out_dtypes(call_node, expected_out_dtype)


@register_func_to_op_list(list_ops=DEFAULT_FOLLOW_LIST)
def generic_follow_op(call_node: "relay.Call", expected_out_dtype: str) -> List:
    return [MIXED_PRECISION_FOLLOW] + get_generic_out_dtypes(call_node, expected_out_dtype)


@register_func_to_op_list(list_ops=DEFAULT_NEVER_LIST)
def generic_never_op(call_node: "relay.Call", expected_out_dtype: str) -> List:
    return [MIXED_PRECISION_NEVER] + get_generic_out_dtypes(call_node, expected_out_dtype)
