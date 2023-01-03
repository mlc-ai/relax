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

import tvm
from tvm import relax
from tvm.script import tir as T, relax as R
from tvm import topi
import numpy as np
from tvm.ir import IRModule
import tvm.testing
import tempfile
import tvm.relax.transform.tuning_api.default_functions
from tvm.relax.transform.tuning_api import Trace


def apply_passes(mod: IRModule, target):
    with tempfile.TemporaryDirectory() as work_dir:
        with target, tvm.transform.PassContext(trace=Trace(mod)):
            seq = tvm.transform.Sequential(
                [
                    relax.transform.MetaScheduleTuneIRMod(
                        params={}, work_dir=work_dir, max_trials_global=8
                    ),
                    relax.transform.MetaScheduleApplyDatabase(work_dir),
                ]
            )
            mod = seq(mod)

    mod = relax.transform.ToNonDataflow()(mod)
    mod = relax.transform.CallTIRRewrite()(mod)
    mod = relax.transform.CanonicalizeBindings()(mod)
    mod = relax.transform.RewriteCUDAGraph()(mod)
    # mod = relax.transform.VMGraphMemoryPlan()(mod)
    mod = relax.transform.VMMemoryLower()(mod)
    mod = relax.transform.VMShapeLower()(mod)
    mod = relax.transform.AttachGlobalSymbol()(mod)
    return mod


def minimal_vm_build(mod, target, params=None):
    # Split primfunc and relax function
    rx_mod, tir_mod = relax.vm._split_tir_relax(mod)
    lib = tvm.build(tir_mod, target=target)

    if params is None:
        params = {}

    return relax.vm.Executable(relax._ffi_api.VMCodeGen(rx_mod, lib, [], target, params))


def test_minimum_example():
    x = relax.Var("x", relax.TensorStructInfo((2, 4)))
    const_one = relax.const(1, "float32")

    bb = relax.BlockBuilder()
    with bb.function("main", [x]):
        v0 = bb.emit_te(topi.exp, x)
        v1 = bb.emit_te(topi.reshape, v0, (8,))
        v2 = bb.emit_te(topi.nn.relu, v1)
        v3 = bb.emit_te(topi.add, v2, const_one)
        v4 = bb.emit_te(topi.nn.pad, v3, pad_before=[1], pad_after=[1], pad_value=1)
        v5 = bb.emit_te(topi.log, v4)
        bb.emit_func_output(v5)

    mod = bb.get()
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    mod = apply_passes(mod, target)

    dev = tvm.cuda(0)
    exec = minimal_vm_build(mod, target)
    vm = relax.VirtualMachine(exec, dev)

    x_np = np.random.rand(2, 4).astype("float32")
    y_np = np.exp(x_np)
    y_np = np.reshape(y_np, (8,))
    y_np = np.maximum(y_np, 0.0)
    y_np = np.add(y_np, 1)
    pad_value = np.ones((1,)).astype("float32")
    y_np = np.concatenate([pad_value, y_np, pad_value], axis=0)
    y_np = np.log(y_np)

    x_relax = tvm.nd.array(x_np, dev)
    y_relax = vm["main"](x_relax)

    tvm.testing.assert_allclose(y_relax.numpy(), y_np, rtol=1e-5, atol=1e-5)
    print("OK")


if __name__ == "__main__":
    test_minimum_example()
