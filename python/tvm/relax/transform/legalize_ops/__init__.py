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
"""Legalize high-level operator calls in Relax functions to call_tir."""
from typing import Optional, Dict
from .. import _ffi_api

from .binary import *
from .creation import *
from .datatype import *
from .image import *
from .index import *
from .linear_algebra import *
from .manipulate import *
from .nn import *
from .search import *
from .statistical import *
from .unary import *


def LegalizeOps(customize_legalize_map: Optional[Dict[str, LegalizeFunc]] = None):
    """Legalize high-level operator calls in Relax functions to call_tir
    with corresponding low-level TIR PrimFuncs.

    For each high-level operator, we register the way of legalizing it as a
    function, which takes a context BlockBuilder and the Call being legalized
    as input, and returns the legalized call. Here the input BlockBuilder is
    mainly used for adding the PrimFunc created by call_te into the context
    IRModule.

    The legalization function for each operator is registered in a map,
    where the operator name is the key. The default legalization functions
    are in the map `DEFAULT_OP_LEGALIZE_MAP`.

    This pass provides customizability for users to use their own legalization
    function for operators. The pass takes an optional customized map,
    which has the same key/value type as the default map (see `LegalizeFunc`),
    from users. When an operator is contained in both the default map and the
    customized map, the default legalization function will be overridden, and
    only the customized one will be used.

    Parameters
    ----------
    customize_legalize_map : Optional[Dict[str, LegalizeFunc]]
        The customized operator legalization function map.
        If not specified, it will be a fresh empty dict.
        If an op has legalization function in both the default map and the
        customized map, the customized function will override the default
        one.

    Examples
    --------
    The following code shows how to use this pass:

    .. code-block:: python

        # Define the pass input IRModule
        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                z: R.Tensor((2, 3), "float32") = R.add(x, y)
                r: R.Tensor((2, 3), "float32") = R.multiply(y, z)
                return r

        # Define the customized legalization function for "relax.add"
        def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
            from tvm import topi
            return bb.call_te(topi.add, call.args[1], call.args[0])

        # Apply the pass with the customized function to the module.
        mod = LegalizeOps({"relax.add": customize_legalize_add})(Module)

    Print out the result by `mod.show()`, we can see the IRModule after
    legalization becomes

    .. code-block:: python

        @tvm.script.ir_module
        class Module:
            @R.function
            def main(
                x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")
            ) -> R.Tensor((2, 3), "float32"):
                z = R.call_tir(add, (y, x), (2, 3), dtype="float32")
                r = R.call_tir(multiply, (y, z), (2, 3), dtype="float32")
                return r

            @T.prim_func
            def add(
                A: T.Buffer[(2, 3), "float32"],
                B: T.Buffer[(2, 3), "float32"],
                T_add: T.Buffer[(2, 3), "float32"],
            ):
                T.func_attr({"tir.noalias": True})
                for ax0, ax1 in T.grid(2, 3):
                    with T.block("T_add"):
                        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                        T.writes(T_add[v_ax0, v_ax1])
                        T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

            @T.prim_func
            def multiply(
                A: T.Buffer[(2, 3), "float32"],
                B: T.Buffer[(2, 3), "float32"],
                T_multiply: T.Buffer[(2, 3), "float32"],
            ):
                T.func_attr({"tir.noalias": True})
                for ax0, ax1 in T.grid(2, 3):
                    with T.block("T_multiply"):
                        v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                        T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                        T.writes(T_multiply[v_ax0, v_ax1])
                        T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[v_ax0, v_ax1]
    """

    return _ffi_api.LegalizeOps(customize_legalize_map)  # type: ignore
