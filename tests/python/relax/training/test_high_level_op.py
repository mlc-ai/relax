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

from utils import LowerToTensorIRPass

# tensor/unary
#   transpose
#   log
#   negative
#   ones_like
#   zeros_like
# tensor/binary
#   matmul
#   collapse_sum_like
# nn/
#   relu
#   gradrelu_
#   softmax
#   dense
#   cross_entropy
#   softmax_cross_entropy
#   sigmoid
#   tanh

def run_relax(op, *input_data):
    """
        generate module and run it to get output
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
            out = bb.emit_output(op(*input_list))
        bb.emit_func_output(out)
    target = tvm.target.Target("llvm")
    bb.get().show()
    mod = LowerToTensorIRPass()(bb.get())
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm["main"](*tvm_data)

def test_transpose():
    data_numpy = np.random.randint(0, 16, (3, 4)).astype(np.float32)
    expected_output = np.transpose(data_numpy)
    result = run_relax(relax.op.transpose, data_numpy)
    np.testing.assert_array_equal(expected_output, result.numpy())

def test_log():
    data_numpy = np.random.randint(1, 16, (16, 16)).astype(np.float32)
    expected_output = np.log(data_numpy)
    result = run_relax(relax.op.log, data_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def test_negative():
    data_numpy = np.random.randint(0, 16, (16, 16)).astype(np.float32)
    expected_output = np.negative(data_numpy)
    result = run_relax(relax.op.negative, data_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def test_ones_like():
    data_numpy = np.zeros((16, 16)).astype(np.float32)
    expected_output = np.ones_like(data_numpy)
    result = run_relax(relax.op.ones_like, data_numpy)
    np.testing.assert_array_equal(expected_output, result.numpy())

def test_zeros_like():
    data_numpy = np.zeros((16, 16)).astype(np.float32)
    expected_output = np.zeros_like(data_numpy)
    result = run_relax(relax.op.zeros_like, data_numpy)
    np.testing.assert_array_equal(expected_output, result.numpy())

def test_matmul():
    data1_numpy = np.random.randint(0, 16, (7, 8)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (8, 10)).astype(np.float32)
    expected_output = np.matmul(data1_numpy, data2_numpy)
    result = run_relax(relax.op.matmul, data1_numpy, data2_numpy)
    np.testing.assert_array_equal(expected_output, result.numpy())

def test_collapse_sum_like():
    data1_numpy = np.random.randint(0, 16, (1, 10)).astype(np.float32)
    data2_numpy = np.zeros((10,)).astype(np.float32)
    expected_output = data1_numpy.reshape((10,))
    assert not np.array_equal(data1_numpy, expected_output)
    result = run_relax(relax.op.collapse_sum_like, data1_numpy, data2_numpy)
    np.testing.assert_array_equal(expected_output, result.numpy())

def test_relu():
    data_numpy = np.random.randint(-16, 16, (16, 16)).astype(np.float32)
    expected_output = np.maximum(data_numpy, 0)
    result = run_relax(relax.op.nn.relu, data_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def test_gradrelu_():
    data_numpy = np.random.randint(-16, 16, (16, 16)).astype(np.float32)
    expected_output = (data_numpy > 0).astype(np.float32)
    result = run_relax(relax.op.nn.gradrelu_, data_numpy)
    np.testing.assert_array_equal(expected_output, result.numpy())

def softmax_numpy(x):
    x = np.exp(x-np.max(x))
    return x / np.sum(x)

def test_softmax():
    data_numpy = np.random.randint(-16, 16, (10,)).astype(np.float32)
    expected_output = softmax_numpy(data_numpy)
    result = run_relax(relax.op.nn.softmax, data_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def test_dense():
    data1_numpy = np.random.randint(0, 16, (3, 4)).astype(np.float32)
    data2_numpy = np.random.randint(0, 16, (3, 4)).astype(np.float32)
    expected_output = np.matmul(data1_numpy, data2_numpy.T)
    result = run_relax(relax.op.nn.dense, data1_numpy, data2_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def cross_entropy_numpy(x, y):
    return np.sum(-np.log(x) * y)

def test_cross_entropy():
    data1_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    data2_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    expected_output = cross_entropy_numpy(data1_numpy, data2_numpy)
    result = run_relax(relax.op.nn.cross_entropy, data1_numpy, data2_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def test_softmax_cross_entropy():
    data1_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    data2_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    expected_output = cross_entropy_numpy(softmax_numpy(data1_numpy), data2_numpy)
    result = run_relax(relax.op.nn.softmax_cross_entropy, data1_numpy, data2_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def test_sum():
    data_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    expected_output = np.sum(data_numpy)
    result = run_relax(relax.op.sum, data_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def sigmoid_numpy(z):
    return 1/(1 + np.exp(-z))

def test_sigmoid():
    data_numpy = np.random.randint(1, 16, (10,)).astype(np.float32)
    expected_output = sigmoid_numpy(data_numpy)
    result = run_relax(relax.op.nn.sigmoid, data_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

def test_tanh():
    data_numpy = np.random.randint(1, 16, (16, 16)).astype(np.float32)
    expected_output = np.tanh(data_numpy)
    result = run_relax(relax.op.nn.tanh, data_numpy)
    np.testing.assert_allclose(expected_output, result.numpy(), rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__])
