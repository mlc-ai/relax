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
"""Attach skip attribute to dispatch to CuBLAS, then dispatch
The pass is written in Python for experiment, fast development.
"""

import tvm
from tvm.ir.module import IRModule
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix


@tvm.transform.module_pass(opt_level=0, name="BLASDispatch")
class BLASDispatch:  # pylint: disable=too-few-public-methods,broad-exception-raised
    """A compiler pass that dispatches patterns to cuBLAS/hipBLAS."""

    def __init__(self, target: tvm.target.Target) -> None:
        if target.kind.name == "cuda":
            self.has_blas = tvm.get_global_func("relax.ext.cublas", True)
            self.patterns = get_patterns_with_prefix("cublas")
        elif target.kind.name == "rocm":
            self.has_blas = tvm.get_global_func("relax.ext.hipblas", True)
            self.patterns = get_patterns_with_prefix("hipblas")

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        
        model_names = []
        for gv, func in mod.functions_items():
            if "relax.backend.blas_dispatch" in func.attrs and func.attrs["relax.backend.blas_dispatch"] is not False:
                model_names.append(gv.name_hint)

        mod = tvm.transform.Sequential(
            [
                tvm.relax.transform.FuseOpsByPattern(
                    self.patterns,
                    bind_constants=False,
                    annotate_codegen=True,
                    entry_functions=model_names,
                ),
                tvm.relax.transform.RunCodegen({}, entry_functions=model_names),
            ]
        )(mod)
        return mod