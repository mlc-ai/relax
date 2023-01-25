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


A_TYPE = "float32"
B_TYPE = "float32"
C_TYPE = "float32"


@register_func("tvm.relax.cublas.get_op_pattern_list")
def get_op_pattern_list():
    return OP_PATTERN_LIST


@register_func("tvm.relax.cublas.get_op_pattern")
def get_op_pattern(name):
    return OP_PATTERN_GENERATOR_LIST[name]()


@register_pattern_generator("dense_row_row_row")
def dense_row_row_row():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    k = tir.Var("k", "int32")
    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name("dense_row_row_row")
                A = T.arg("A", T.buffer_decl((m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
                with T.grid(m, n, k) as (l0, l1, l2):
                    with T.block("dense_row_row_row"):
                        vi, vj, vk = T.axis.remap("SSR", [l0, l1, l2])
                        T.reads(A[vi, vk], B[vk, vj])
                        T.writes(C[vi, vj])
                        with T.init():
                            T.buffer_store(C, T.cast(0.0, C_TYPE), [vi, vj])
                        T.buffer_store(C, C[vi, vj] + A[vi, vk] * B[vk, vj], [vi, vj])
    return ib.get()["dense_row_row_row"]


@register_func("tvm.relax.cublas.get_graph_pattern_code")
def get_graph_pattern_code(cublas_op):
    cublas_op = [str(st) for st in cublas_op]
    pattern = "/".join(cublas_op)
    if pattern not in GRAPH_PATTERN_CODE_LIST:
        raise tvm.TVMError("Cannot find graph pattern code for cublas op: {}".format(cublas_op))
    return GRAPH_PATTERN_CODE_LIST["/".join(cublas_op)]


GRAPH_PATTERN_CODE_LIST[
    "dense_row_row_row"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1

      #include <cublas_v2.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm::runtime;

      void _GEMM(NDArray A, NDArray B, NDArray C) {
        // A: [M, K], B: [K, N]
        CHECK_EQ(A->ndim, 2);
        CHECK_EQ(B->ndim, 2);
        CHECK_EQ(C->ndim, 2);
        CHECK_EQ(A->shape[1], B->shape[0]);
        int M = A->shape[0];
        int K = A->shape[1];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], M);
        CHECK_EQ(C->shape[1], N);
        // Define variables
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        int lda = K;
        int ldb = N;
        int ldc = N;
        float* a = reinterpret_cast<float*>(A->data);
        float* b = reinterpret_cast<float*>(B->data);
        float* c = reinterpret_cast<float*>(C->data);
        // Set up Cublas handle
        cublasHandle_t handle;
        cublasCreate(&handle);
        // Cublas status
        cudaError_t cudaStatus;
        // Cublas is column primary, so the order is transposed
        cublasSgemm(
            handle,            //
            CUBLAS_OP_N,       //
            CUBLAS_OP_N,       //
            N,                 //
            M,                 //
            K,                 //
            &alpha,            //
            b,                 //
            ldb,               //
            a,                 //
            lda,               //
            &beta,             //
            c,                 //
            ldc                //
        );
        cudaThreadSynchronize();
        cudaStatus = cudaGetLastError();
        CHECK(cudaStatus == cudaSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _GEMM);
      """
