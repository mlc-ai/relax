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
"""Pre-defined pipelines.

oRelax enables flexible pipeline optimizations before min build.
This namespace offers a pre-defined collection that can be used
as it is or serves as a basis to do further composition.
"""
# pylint: disable=unused-argument
import tvm
from tvm import meta_schedule as ms
from . import transform


@tvm.transform.module_pass(opt_level=0)
def zero_pipeline(mod: tvm.ir.IRModule, ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
    """Pipeline that applies pre-tuned logs.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        Input IRModule.

    ctx : tvm.transform.PassContext
        The pass context

    Returns
    -------
    mod: tvm.ir.IRModule
        The result transformed module.
    """
    seq = tvm.transform.Sequential(
        [
            transform.LegalizeOps(),
            transform.AnnotateTIROpPattern(),
            transform.FoldConstant(),
            transform.FuseOps(),
            transform.FuseTIR(),
        ]
    )
    mod = seq(mod)
    if ms.Database.current():
        mod = transform.MetaScheduleApplyDatabase()(mod)
    return mod


# global map of pre-built pipelines
PIPELINE_MAP = {"zero": zero_pipeline}


def get_pipeline(name: str = "zero") -> tvm.transform.Pass:
    """Get pre-build pipeline by name

    Parameters
    ----------
    name : Optional[str]
        Name of the pipeline

    Returns
    -------
    pipeline: tvm.transform.Pass
       The transformation pipeline.
    """

    if name in PIPELINE_MAP:
        return PIPELINE_MAP[name]
    else:
        raise ValueError(
            f"Uknown pre-built pipeline {name}," f"candidates are {list(PIPELINE_MAP.keys())}"
        )
