import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
import torchvision
import tvm
import tvm.testing

from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tvm import topi, relax, te
from tvm.script import tir as T


batch_size = 4
input_shape = (batch_size, 1, 28, 28)
weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))


def create_model():
    bb = relax.BlockBuilder()
    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")

    x = relax.Var("x", input_shape, relax.DynTensorType(batch_size, "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.conv2d(x, conv2d_weight, (3, 3)))
            lv1 = bb.emit(relax.op.add(lv0, conv2d_bias))
            lv2 = bb.emit(relax.op.nn.relu(lv1))
            lv3 = bb.emit(relax.op.nn.max_pool2d(lv2, (2, 2)))
            lv4 = bb.emit(relax.op.nn.flatten(lv3))
            lv5 = bb.emit(relax.op.nn.dense(lv4, linear0_weight))
            lv6 = bb.emit(relax.op.add(lv5, linear0_bias))
            lv7 = bb.emit(relax.op.nn.relu(lv6))
            lv8 = bb.emit(relax.op.nn.dense(lv7, linear1_weight))
            lv9 = bb.emit(relax.op.add(lv8, linear1_bias))
            lv10 = bb.emit(relax.op.nn.softmax(lv9))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)
    return bb.get()


print(create_model().script())
