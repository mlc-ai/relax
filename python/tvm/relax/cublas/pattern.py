import tvm
from tvm import register_func, tir
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import tir as T

OP_PATTERN_LIST = list()
OP_PATTERN_GENERATOR_LIST = dict()
GRAPH_PATTERN_CODE_LIST = dict()


def register_pattern_generator(name):
    def register(func):
        OP_PATTERN_LIST.append(name)
        OP_PATTERN_GENERATOR_LIST[name] = func
        return func

    return register
