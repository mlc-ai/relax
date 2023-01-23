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


@register_func("tvm.relax.cublas.op_pattern_stitch")
def op_pattern_stitch(evaluated_symbols, evaluated_buffers, matched_pattern_names):
    return matched_pattern_names


A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


@register_func("tvm.relax.cublas.get_op_pattern_list")
def get_op_pattern_list():
    return OP_PATTERN_LIST


@register_func("tvm.relax.cublas.get_op_pattern")
def get_op_pattern(name):
    return OP_PATTERN_GENERATOR_LIST[name]()


@register_func("tvm.relax.cublas.get_graph_pattern_code")
def get_graph_pattern_code(cublas_op):
    cublas_op = [str(st) for st in cublas_op]
    pattern = "/".join(cublas_op)
    if pattern not in GRAPH_PATTERN_CODE_LIST:
        raise tvm.TVMError("Cannot find graph pattern code for cublas op: {}".format(cublas_op))
    return GRAPH_PATTERN_CODE_LIST["/".join(cublas_op)]
