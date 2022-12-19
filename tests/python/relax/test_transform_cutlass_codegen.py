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

from __future__ import annotations
import tempfile

from tvm import relax, runtime
import tvm
import tvm.testing
from tvm import relax
import numpy as np
from tvm.relax.vm import build as relax_build
from tvm.relax.transform import OperatorLegalizer
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tir as T
from tvm.script.ir_builder import IRBuilder

import tvm.relax.cutlass.pattern

PKG_FILE = "/tmp/test_transform_cutlass_codegen.so"
GLOBAL_SYMBOL = "HGEMM"
A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"

target = "cuda"


def f_run(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    return vm["main"](*input)


def build(mod):
    mod = relax.transform.OperatorLegalizer(mod).transform()
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.SplitCutlass()(mod)
    executbale = relax_build(mod, target)
    executbale.mod.export_library(PKG_FILE, cc="nvcc")
    return executbale


def constructGEMM(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.nn.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


def test_cutlass_dense():
    m, n, k = 128, 128, 128
    build(constructGEMM(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructGEMM_bias(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, N), A_TYPE))  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.nn.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


def test_cutlass_dense_bias():
    m, n, k = 128, 128, 128
    build(constructGEMM_bias(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructGEMM_bias_relu(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, N), A_TYPE))  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.nn.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.nn.relu(D))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    relax_mod = ib.get()
    return relax_mod


def test_cutlass_dense_bias_relu():
    m, n, k = 128, 128, 128
    build(constructGEMM_bias_relu(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), np.maximum(A @ B + bias, 0), rtol=1e-2)


def constructBatchGEMM(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((batch, M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.nn.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


def test_cutlass_batch_dense():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM2(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((batch, M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((batch, K, N), B_TYPE))  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.nn.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


def test_cutlass_batch_dense2():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM2(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((batch, M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, N), A_TYPE))  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.nn.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


def test_cutlass_batch_dense_bias():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructBatchGEMM2_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((batch, M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((batch, K, N), B_TYPE))  # pylint: disable=invalid-name
                bias = R.arg("bias", R.tensor((1, N), A_TYPE))  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.nn.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


def test_cutlass_batch_dense2_bias():
    b, m, n, k = 2, 128, 128, 128
    build(constructBatchGEMM2_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructConv2D(N, C, H, W, KH, KW, O, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg("x", R.tensor((N, H, W, C), A_TYPE))  # pylint: disable=invalid-name
                w = R.arg("w", R.tensor((O, KH, KW, C), B_TYPE))  # pylint: disable=invalid-name
                C = R.nn.conv2d(
                    x,
                    w,
                    kernel_size=(KH, KW),
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    groups=1,
                    channels=None,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype=C_TYPE,
                )
                R.func_ret_value(C)
    mod = ib.get()
    return mod


def test_cutlass_conv2d():
    n, c, h, w = 1, 3, 224, 224
    kh, kw, o = 3, 3, 64
    strides = (2, 2)
    padding = (3, 3)
    dilation = (4, 4)
    build(constructConv2D(n, c, h, w, kh, kw, o, strides, padding, dilation))
    dev = tvm.cuda()
    np.random.seed(0)
    A = np.random.rand(n, h, w, c).astype("float16") * 5
    B = np.random.rand(o, kh, kw, c).astype("float16") * 5
    print(A.shape, B.shape)
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    executable = tvm.runtime.load_module(PKG_FILE)
    result = f_run(executable, dev, A_tvm, B_tvm)


if __name__ == "__main__":
    # test_cutlass_dense()
    # test_cutlass_dense_bias()
    # test_cutlass_dense_bias_relu()
    # test_cutlass_batch_dense()
    # test_cutlass_batch_dense2()
    # test_cutlass_batch_dense_bias()
    # test_cutlass_batch_dense2_bias()
    test_cutlass_conv2d()
