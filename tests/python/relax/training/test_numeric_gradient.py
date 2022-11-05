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
import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.testing.utils import check_numerical_grads

from utils import LowerToTensorIRPass
import _gradient

# add
# sub
# multiply
# transpose
# nn.relu
# nn.matmul
# nn.softmax_cross_entropy
# nn.sigmoid
# nn.tanh

def relax_check_gradients(op, input_data, is_output_scalar = False):
    """
        generate module, transform it with SimpleADD and run it to check numberic gradients
        ---
        input: numpy array
    """
    # print(op, type(op))
    func_name = "main"
    data_type = relax.DynTensorType(dtype="float32")
    input_list = []
    tvm_data = []

    for i in range(len(input_data)):
        input_list.append(relax.Var("x_" + str(i), input_data[i].shape, data_type))
        tvm_data.append(tvm.nd.array(input_data[i]))

    bb = relax.BlockBuilder()
    with bb.function(func_name, input_list):
        with bb.dataflow():
            if is_output_scalar:
                out = bb.emit_output(op(*input_list))
            else: 
                lv0 = bb.emit(op(*input_list))
                out = bb.emit_output(relax.op.sum(lv0))
        bb.emit_func_output(out)
    target = tvm.target.Target("llvm")
    mod = bb.get()
    mod.show()
    lower_mod = LowerToTensorIRPass()(mod)
    
    def forward(*inputs):
        ex_0 = relax.vm.build(lower_mod, target)
        vm_0 = relax.VirtualMachine(ex_0, tvm.cpu())
        result = vm_0["main"](*[tvm.nd.array(i) for i in inputs])
        return result.numpy()

    ad_mod = relax.transform.SimpleAD(mod.get_global_var("main"))(mod)
    lower_ad_mod = LowerToTensorIRPass()(ad_mod)
    ex = relax.vm.build(lower_ad_mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    _, grads = vm["main_adjoint"](*tvm_data)

    check_numerical_grads(forward, input_data, [i.numpy() for i in grads])


def test_add():
    data1_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.add, [data1_numpy, data2_numpy])

def test_sub():
    data1_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.sub, [data1_numpy, data2_numpy])

def test_transpose():
    data1_numpy = np.random.randint(0, 16, (5, 10)).astype(np.float32)
    relax_check_gradients(relax.op.transpose, [data1_numpy])

def test_relu():
    data1_numpy = np.random.uniform(-1, 1, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.nn.relu, [data1_numpy])

def test_matmul():
    data1_numpy = np.random.randint(0, 16, (7, 8)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (8, 10)).astype(np.float32)
    relax_check_gradients(relax.op.nn.matmul, [data1_numpy, data2_numpy])

def test_softmax_cross_entropy():
    data1_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    data2_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    data2_numpy /= np.sum(data2_numpy)
    relax_check_gradients(relax.op.nn.softmax_cross_entropy, [data1_numpy, data2_numpy], True)

def test_sigmoid():
    data_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    relax_check_gradients(relax.op.nn.sigmoid, [data_numpy])

def test_tanh():
    data_numpy = np.random.randint(1, 16, (16, 16)).astype(np.float32)
    relax_check_gradients(relax.op.nn.tanh, [data_numpy])

if __name__ == "__main__":
    pytest.main([__file__])
    # test_softmax_cross_entropy()
    # test_sigmoid()
    # test_tanh()