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
import tempfile

import numpy as np
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relax, topi
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.relax.testing import relay_translator
from tvm.script.parser import relax as R


# pylint: disable=no-member,line-too-long,too-many-nested-blocks,unbalanced-tuple-unpacking,no-self-argument,missing-docstring,invalid-name


@tvm.testing.requires_package("torch")
def test_task_extraction_anchor_block():
    mod, params, _ = get_network(name="resnet_18", input_shape=[1, 3, 224, 224])
    relax_mod = relay_translator.from_relay(mod["main"], "llvm", params)
    relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
    relax_mod = relax.transform.FuseOps()(relax_mod)
    relax_mod = relax.transform.FuseTIR()(relax_mod)
    extracted_tasks = ms.relax_integration.extract_tasks(
        relax_mod, target="llvm", params=params, module_equality="anchor-block"
    )
    # Note that there is no task from residual blocks
    expected_task_names = [
        "layout_transform",
        "fused_conv2d_add_relu",
        "max_pool2d",
        "fused_conv2d1_add1_relu1",
        "fused_conv2d2_add3_relu2",
        "fused_conv2d4_add3",
        "fused_conv2d3_add3_add4_relu2",
        "fused_conv2d5_add5_relu3",
        "fused_conv2d7_add5",
        "fused_conv2d6_add5_add6_relu3",
        "fused_conv2d8_add7_relu4",
        "fused_conv2d10_add7",
        "fused_conv2d9_add7_add8_relu4",
        "adaptive_avg_pool2d",
        "fused_layout_transform1_reshape_squeeze",
        "fused_dense_add9",
    ]

    assert len(extracted_tasks) == len(expected_task_names)
    for t in extracted_tasks:
        assert t.task_name in expected_task_names, t.task_name


def _test_anchor_tuning(target):
    data_shape = (128, 128)
    weight_shape1 = (128, 128)
    weight_shape2 = (128, 128)

    data = relax.Var("data", R.Tensor(data_shape, "float32"))
    weight1 = relax.Var("weight1", R.Tensor(weight_shape1, "float32"))
    weight2 = relax.Var("weight2", R.Tensor(weight_shape2, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", (data, weight1, weight2)):
        with bb.dataflow():
            dense1 = bb.emit_te(topi.nn.dense, data, weight1)
            add1 = bb.emit_te(topi.add, dense1, relax.const(1.0))
            dense2 = bb.emit_te(topi.nn.dense, add1, weight2)
            lv3 = bb.emit_te(topi.subtract, dense2, data)
            lv4 = bb.emit_te(topi.add, lv3, relax.const(1.0))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    relax_mod = bb.get()
    relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
    relax_mod = relax.transform.FuseOps()(relax_mod)
    relax_mod = relax.transform.FuseTIR()(relax_mod)

    weight1_np = np.random.randn(*weight_shape1).astype("float32")
    weight2_np = np.random.randn(*weight_shape2).astype("float32")

    data_np = np.random.randn(*data_shape).astype("float32")
    params = {"weight1": weight1_np, "weight2": weight2_np}

    module_equality = "anchor-block"

    extracted_tasks = ms.relax_integration.extract_tasks(
        relax_mod, target, params, module_equality=module_equality
    )

    assert len(extracted_tasks) == 1

    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.relax_integration.tune_relax(
            mod=relax_mod,
            target=target,
            params=params,
            work_dir=work_dir,
            max_trials_global=4,
            strategy="replay-trace",
            module_equality=module_equality,
        )
        lib = ms.relax_integration.compile_relax(
            database, relax_mod, target, params, module_equality
        )

    dev = tvm.device(target, 0)
    data_tvm = tvm.nd.array(data_np, device=dev)
    vm = relax.VirtualMachine(lib, dev)
    res = vm["main"](data_tvm)
    out = res.numpy()

    from tvm.relax.vm import build as relax_build

    relax_mod = relax.transform.BindParams("main", params)(relax_mod)
    lib_ref = relax_build(relax_mod, target)
    vm_ref = relax.VirtualMachine(lib_ref, dev)
    res_ref = vm_ref["main"](data_tvm)
    ref = res_ref.numpy()

    np.testing.assert_allclose(ref, out, atol=1e-3)


def test_anchor_tuning_cpu():
    _test_anchor_tuning("llvm --num-cores=4")


if __name__ == "__main__":
    tvm.testing.main()
