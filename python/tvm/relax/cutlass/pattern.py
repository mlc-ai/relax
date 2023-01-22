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


@register_func("tvm.relax.cutlass.op_pattern_stitch")
def op_pattern_stitch(evaluated_symbols, evaluated_buffers, matched_pattern_names):
    if len(matched_pattern_names) >= 3:
        assert len(evaluated_symbols) >= 3
        assert len(evaluated_buffers) >= 3
        if (
            matched_pattern_names[0] == "dense_row_row_row"
            and matched_pattern_names[1] == "bias_row"
            and matched_pattern_names[2] == "relu"
        ):
            # dense_row_row_row + bias_row + relu
            m_dense, n_dense, k_dense = evaluated_symbols[0]
            m_bias, n_bias = evaluated_symbols[1]
            m_relu, n_relu = evaluated_symbols[2]
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            A_relu, B_relu = evaluated_buffers[2]
            if (
                m_dense == m_bias
                and n_dense == n_bias
                and m_dense == m_relu
                and n_dense == n_relu
                and C_dense == A_bias
                and C_bias == A_relu
            ):
                return matched_pattern_names[:3]
    if len(matched_pattern_names) >= 2:
        assert len(evaluated_symbols) >= 2
        assert len(evaluated_buffers) >= 2
        if (
            matched_pattern_names[0] == "dense_row_row_row"
            and matched_pattern_names[1] == "bias_row"
        ):
            # dense_row_row_row + bias_row
            m_dense, n_dense, k_dense = evaluated_symbols[0]
            m_bias, n_bias = evaluated_symbols[1]
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                return matched_pattern_names[:2]
        if (
            matched_pattern_names[0] == "batch_dense_row_row_row"
            and matched_pattern_names[1] == "batch_bias_row"
        ):
            # dense_row_row_row + bias_row
            b_dense, m_dense, n_dense, k_dense = evaluated_symbols[0]
            b_bias, m_bias, n_bias = evaluated_symbols[1]
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if b_dense == b_bias and m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                return matched_pattern_names[:2]
        if (
            matched_pattern_names[0] == "batch_dense_row_row_row_2"
            and matched_pattern_names[1] == "batch_bias_row"
        ):
            # dense_row_row_row + bias_row
            b_dense, m_dense, n_dense, k_dense = evaluated_symbols[0]
            b_bias, m_bias, n_bias = evaluated_symbols[1]
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if b_dense == b_bias and m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                return matched_pattern_names[:2]
    if len(matched_pattern_names) >= 1:
        assert len(evaluated_symbols) >= 1
        assert len(evaluated_buffers) >= 1
        if matched_pattern_names[0] == "dense_row_row_row":
            # dense_row_row_row
            return matched_pattern_names[:1]
        elif matched_pattern_names[0] == "batch_dense_row_row_row":
            # batch_dense_row_row_row
            return matched_pattern_names[:1]
        elif matched_pattern_names[0] == "batch_dense_row_row_row_2":
            # batch_dense_row_row_row_2
            return matched_pattern_names[:1]
    return []


A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


@register_func("tvm.relax.cutlass.get_op_pattern_list")
def get_op_pattern_list():
    return OP_PATTERN_LIST


@register_func("tvm.relax.cutlass.get_op_pattern")
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


@register_pattern_generator("bias_row")
def bias_row():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name("bias_row")
                A = T.arg("A", T.buffer_decl((m, n), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((0, n), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
                with T.grid(m, n) as (l0, l1):
                    with T.block("bias_row"):
                        i, j = T.axis.remap("SS", [l0, l1])
                        T.reads(A[i, j], B[0, j])
                        T.writes(C[i, j])
                        T.buffer_store(C, A[i, j] + B[0, j], [i, j])
    return ib.get()["bias_row"]


@register_pattern_generator("batch_bias_row")
def batch_bias_row():
    batch = tir.Var("b", "int32")
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name("batch_bias_row")
                A = T.arg("A", T.buffer_decl((batch, m, n), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((0, n), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((batch, m, n), C_TYPE))  # pylint: disable=invalid-name
                with T.grid(batch, m, n) as (lb, l0, l1):
                    with T.block("batch_bias_row"):
                        b, i, j = T.axis.remap("SSS", [lb, l0, l1])
                        T.reads(A[b, i, j], B[0, j])
                        T.writes(C[b, i, j])
                        T.buffer_store(C, A[b, i, j] + B[0, j], [b, i, j])
    return ib.get()["batch_bias_row"]


@register_pattern_generator("relu")
def relu():
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name("relu")
                A = T.arg("A", T.buffer_decl((m, n), A_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((m, n), C_TYPE))  # pylint: disable=invalid-name
                with T.grid(m, n) as (l0, l1):
                    with T.block("relu"):
                        i, j = T.axis.remap("SS", [l0, l1])
                        T.reads(A[i, j])
                        T.writes(C[i, j])
                        T.buffer_store(C, T.max(A[i, j], T.cast(0, A_TYPE)), [i, j])
    return ib.get()["relu"]


@register_pattern_generator("batch_dense_row_row_row")
def batch_dense_row_row_row():
    b = tir.Var("b", "int32")
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    k = tir.Var("k", "int32")
    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name("batch_dense_row_row_row")
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((k, n), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((b, m, n), C_TYPE))  # pylint: disable=invalid-name
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("batch_dense_row_row_row"):
                        vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vk, vj])
                        T.writes(C[vb, vi, vj])
                        with T.init():
                            T.buffer_store(C, T.cast(0.0, C_TYPE), [vb, vi, vj])
                        T.buffer_store(C, C[vb, vi, vj] + A[vb, vi, vk] * B[vk, vj], [vb, vi, vj])
    return ib.get()["batch_dense_row_row_row"]


@register_pattern_generator("batch_dense_row_row_row_2")
def batch_dense_row_row_row_2():
    b = tir.Var("b", "int32")
    m = tir.Var("m", "int32")
    n = tir.Var("n", "int32")
    k = tir.Var("k", "int32")
    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                T.func_name("batch_dense_row_row_row_2")
                A = T.arg("A", T.buffer_decl((b, m, k), A_TYPE))  # pylint: disable=invalid-name
                B = T.arg("B", T.buffer_decl((b, k, n), B_TYPE))  # pylint: disable=invalid-name
                C = T.arg("C", T.buffer_decl((b, m, n), C_TYPE))  # pylint: disable=invalid-name
                with T.grid(b, m, n, k) as (lb, l0, l1, l2):
                    with T.block("batch_dense_row_row_row_2"):
                        vb, vi, vj, vk = T.axis.remap("SSSR", [lb, l0, l1, l2])
                        T.reads(A[vb, vi, vk], B[vb, vk, vj])
                        T.writes(C[vb, vi, vj])
                        with T.init():
                            T.buffer_store(C, T.cast(0.0, C_TYPE), [vb, vi, vj])
                        T.buffer_store(
                            C, C[vb, vi, vj] + A[vb, vi, vk] * B[vb, vk, vj], [vb, vi, vj]
                        )
    return ib.get()["batch_dense_row_row_row_2"]


@register_func("tvm.relax.cutlass.get_graph_pattern_code")
def get_graph_pattern_code(cutlass_op):
    cutlass_op = [str(st) for st in cutlass_op]
    pattern = "/".join(cutlass_op)
    if pattern not in GRAPH_PATTERN_CODE_LIST:
        raise tvm.TVMError("Cannot find graph pattern code for cutlass op: {}".format(cutlass_op))
    return GRAPH_PATTERN_CODE_LIST["/".join(cutlass_op)]


GRAPH_PATTERN_CODE_LIST[
    "dense_row_row_row"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm;
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
        CHECK_EQ(A.DataType(), DataType::Float(16));
        CHECK_EQ(B.DataType(), DataType::Float(16));
        CHECK_EQ(C.DataType(), DataType::Float(16));
        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::Gemm<cutlass::half_t,            // ElementA
                                                cutlass::layout::RowMajor,  // LayoutA
                                                cutlass::half_t,            // ElementB
                                                cutlass::layout::RowMajor,  // LayoutB
                                                cutlass::half_t,            // ElementOutput
                                                cutlass::layout::RowMajor   // LayoutOutput
                                                >;
        Gemm gemm_op;
        cutlass::half_t alpha(1.0);
        cutlass::half_t beta(0.0);
        cutlass::layout::ColumnMajor::Stride::Index lda(K);
        cutlass::layout::ColumnMajor::Stride::Index ldb(N);
        cutlass::layout::ColumnMajor::Stride::Index ldc(N);
        cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
        cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
        cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::Status status = gemm_op({
            {M, N, K},     //
            {a, lda},      //
            {b, ldb},      //
            {c, ldc},      //
            {c, ldc},      //
            {alpha, beta}  //
        });
        CHECK(status == cutlass::Status::kSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _GEMM);
      """

GRAPH_PATTERN_CODE_LIST[
    "dense_row_row_row/bias_row"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm;
      using namespace tvm::runtime;

      // simple specialized impl, can be replaced by
      // call into libraries.
      void _HGEMM_BIAS(NDArray A, NDArray B, NDArray Bias, NDArray C) {
        // A: [M, K], B: [K, N], BIAS: [1, N], C: [M, N]
        CHECK_EQ(A->ndim, 2);
        CHECK_EQ(B->ndim, 2);
        CHECK_EQ(Bias->ndim, 2);
        CHECK_EQ(C->ndim, 2);
        CHECK_EQ(A->shape[1], B->shape[0]);
        int M = A->shape[0];
        int K = A->shape[1];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], M);
        CHECK_EQ(C->shape[1], N);
        CHECK_EQ(Bias->shape[0], 1);
        CHECK_EQ(Bias->shape[1], N);
        CHECK_EQ(A.DataType(), DataType::Float(16));
        CHECK_EQ(B.DataType(), DataType::Float(16));
        CHECK_EQ(Bias.DataType(), DataType::Float(16));
        CHECK_EQ(C.DataType(), DataType::Float(16));

        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::Gemm<
          cutlass::half_t, cutlass::layout::RowMajor,
          cutlass::half_t, cutlass::layout::RowMajor,
          cutlass::half_t, cutlass::layout::RowMajor,
          cutlass::half_t,
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm75,
          cutlass::gemm::GemmShape<64, 64, 32>,
          cutlass::gemm::GemmShape<32, 32, 32>,
          cutlass::gemm::GemmShape<16, 8, 8>,

          cutlass::epilogue::thread::LinearCombination<
            cutlass::half_t,
            8,
            cutlass::half_t,
            cutlass::half_t,
            cutlass::epilogue::thread::ScaleType::NoBetaScaling
          >,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
          2,
          8,
          8,
          false,
          cutlass::arch::OpMultiplyAdd
        >;

        Gemm gemm_op;

        cutlass::half_t alpha(1.0);
        cutlass::half_t beta(0.0);
        cutlass::layout::ColumnMajor::Stride::Index lda(K);
        cutlass::layout::ColumnMajor::Stride::Index ldb(N);
        cutlass::layout::ColumnMajor::Stride::Index ld_bias(0);
        cutlass::layout::ColumnMajor::Stride::Index ldc(N);
        cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
        cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
        cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);

        cutlass::Status status = gemm_op({
            {M, N, K},            // GemmCoord problem_size_
            {a, lda},             // TensorRef<ElementA const, LayoutA> ref_A_
            {b, ldb},             // TensorRef<ElementB const, LayoutB> ref_B_
            {bias, ld_bias},      // TensorRef<ElementC const, LayoutC> ref_C_
            {c, ldc},             // TensorRef<ElementC, LayoutC> ref_D_
            {alpha, beta}         // typename EpilogueOutputOp::Params epilogue_
        });
        CHECK(status == cutlass::Status::kSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _HGEMM_BIAS);
"""

GRAPH_PATTERN_CODE_LIST[
    "dense_row_row_row/bias_row/relu"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm;
      using namespace tvm::runtime;

      // simple specialized impl, can be replaced by
      // call into libraries.
      void _HGEMM_BIAS_RELU(NDArray A, NDArray B, NDArray Bias, NDArray C) {
        // A: [M, K], B: [K, N], BIAS: [1, N], C: [M, N]
        CHECK_EQ(A->ndim, 2);
        CHECK_EQ(B->ndim, 2);
        CHECK_EQ(Bias->ndim, 2);
        CHECK_EQ(C->ndim, 2);
        CHECK_EQ(A->shape[1], B->shape[0]);
        int M = A->shape[0];
        int K = A->shape[1];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], M);
        CHECK_EQ(C->shape[1], N);
        CHECK_EQ(Bias->shape[0], 1);
        CHECK_EQ(Bias->shape[1], N);
        CHECK_EQ(A.DataType(), DataType::Float(16));
        CHECK_EQ(B.DataType(), DataType::Float(16));
        CHECK_EQ(Bias.DataType(), DataType::Float(16));
        CHECK_EQ(C.DataType(), DataType::Float(16));

        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::Gemm<
          cutlass::half_t, cutlass::layout::RowMajor,
          cutlass::half_t, cutlass::layout::RowMajor,
          cutlass::half_t, cutlass::layout::RowMajor,
          cutlass::half_t,
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm75,
          cutlass::gemm::GemmShape<64, 64, 32>,
          cutlass::gemm::GemmShape<32, 32, 32>,
          cutlass::gemm::GemmShape<16, 8, 8>,

          cutlass::epilogue::thread::LinearCombinationRelu<
            cutlass::half_t,
            8,
            cutlass::half_t,
            cutlass::half_t,
            cutlass::epilogue::thread::ScaleType::NoBetaScaling
          >,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
          2,
          8,
          8,
          false,
          cutlass::arch::OpMultiplyAdd
        >;

        Gemm gemm_op;

        cutlass::half_t alpha(1.0);
        cutlass::half_t beta(0.0);
        cutlass::layout::ColumnMajor::Stride::Index lda(K);
        cutlass::layout::ColumnMajor::Stride::Index ldb(N);
        cutlass::layout::ColumnMajor::Stride::Index ld_bias(0);
        cutlass::layout::ColumnMajor::Stride::Index ldc(N);
        cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
        cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
        cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);

        cutlass::Status status = gemm_op({
            {M, N, K},            // GemmCoord problem_size_
            {a, lda},             // TensorRef<ElementA const, LayoutA> ref_A_
            {b, ldb},             // TensorRef<ElementB const, LayoutB> ref_B_
            {bias, ld_bias},      // TensorRef<ElementC const, LayoutC> ref_C_
            {c, ldc},             // TensorRef<ElementC, LayoutC> ref_D_
            {alpha, beta}         // typename EpilogueOutputOp::Params epilogue_
        });
        CHECK(status == cutlass::Status::kSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _HGEMM_BIAS_RELU);
    """

GRAPH_PATTERN_CODE_LIST[
    "batch_dense_row_row_row"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm_batched.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm;
      using namespace tvm::runtime;

      // simple specialized impl, can be replaced by
      // call into libraries.
      void _BHGEMM(NDArray A, NDArray B, NDArray C) {
        // A: [Batch, M, K], B: [K, N], C: [Batch, M, N]
        CHECK_EQ(A->ndim, 3);
        CHECK_EQ(B->ndim, 2);
        CHECK_EQ(C->ndim, 3);
        CHECK_EQ(A->shape[2], B->shape[0]);
        int Batch = A->shape[0];
        int M = A->shape[1];
        int K = A->shape[2];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], Batch);
        CHECK_EQ(C->shape[1], M);
        CHECK_EQ(C->shape[2], N);
        CHECK_EQ(A.DataType(), DataType::Float(16));
        CHECK_EQ(B.DataType(), DataType::Float(16));
        CHECK_EQ(C.DataType(), DataType::Float(16));

        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::GemmBatched<cutlass::half_t,            // ElementA
                                                        cutlass::layout::RowMajor,  // LayoutA
                                                        cutlass::half_t,            // ElementB
                                                        cutlass::layout::RowMajor,  // LayoutB
                                                        cutlass::half_t,            // ElementOutput
                                                        cutlass::layout::RowMajor   // LayoutOutput
                                                        >;
        Gemm gemm_op;
        cutlass::half_t alpha(1.0);
        cutlass::half_t beta(0.0);
        cutlass::layout::ColumnMajor::Stride::Index lda(K);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_A(M * K);
        cutlass::layout::ColumnMajor::Stride::Index ldb(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_B(0);
        cutlass::layout::ColumnMajor::Stride::Index ldc(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_C(M * N);
        cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
        cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
        cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::Status status = gemm_op({
            {M, N, K},        //
            {a, lda},         //
            batch_stride_A,   //
            {b, ldb},         //
            batch_stride_B,   //
            {c, ldc},         //
            batch_stride_C,   //
            {c, ldc},         //
            batch_stride_C,   //
            {alpha, beta},    //
            Batch             //
        });
        CHECK(status == cutlass::Status::kSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _BHGEMM); 
"""

GRAPH_PATTERN_CODE_LIST[
    "batch_dense_row_row_row/batch_bias_row"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm_batched.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm;
      using namespace tvm::runtime;

      // simple specialized impl, can be replaced by
      // call into libraries.
      void _BHGEMM(NDArray A, NDArray B, NDArray Bias, NDArray C) {
        // A: [Batch, M, K], B: [K, N], Bias: [1, N], C: [Batch, M, N]
        CHECK_EQ(A->ndim, 3);
        CHECK_EQ(B->ndim, 2);
        CHECK_EQ(Bias->ndim, 2);
        CHECK_EQ(C->ndim, 3);
        CHECK_EQ(A->shape[2], B->shape[0]);
        CHECK_EQ(Bias->shape[0], 1);
        CHECK_EQ(Bias->shape[1], B->shape[1]);
        int Batch = A->shape[0];
        int M = A->shape[1];
        int K = A->shape[2];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], Batch);
        CHECK_EQ(C->shape[1], M);
        CHECK_EQ(C->shape[2], N);
        CHECK_EQ(A.DataType(), DataType::Float(16));
        CHECK_EQ(B.DataType(), DataType::Float(16));
        CHECK_EQ(C.DataType(), DataType::Float(16));

        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::GemmBatched<
            cutlass::half_t,            // ElementA
            cutlass::layout::RowMajor,  // LayoutA
            cutlass::half_t,            // ElementB
            cutlass::layout::RowMajor,  // LayoutB
            cutlass::half_t,            // ElementOutput
            cutlass::layout::RowMajor,  // LayoutOutput
            cutlass::half_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm75
        >;
        Gemm gemm_op;
        cutlass::half_t alpha(1.0);
        cutlass::half_t beta(0.0);
        cutlass::layout::ColumnMajor::Stride::Index lda(K);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_A(M * K);
        cutlass::layout::ColumnMajor::Stride::Index ldb(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_B(0);
        cutlass::layout::ColumnMajor::Stride::Index ld_bias(0);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_bias(0);
        cutlass::layout::ColumnMajor::Stride::Index ldc(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_C(M * N);
        cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
        cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
        cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);
        cutlass::Status status = gemm_op({
            {M, N, K},           //
            {a, lda},            //
            batch_stride_A,      //
            {b, ldb},            //
            batch_stride_B,      //
            {bias, ld_bias},     //
            batch_stride_bias,   //
            {c, ldc},            //
            batch_stride_C,      //
            {alpha, beta},       //
            Batch                //
        });
        CHECK(status == cutlass::Status::kSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _BHGEMM); 
"""

GRAPH_PATTERN_CODE_LIST[
    "batch_dense_row_row_row_2"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm_batched.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm;
      using namespace tvm::runtime;

      // simple specialized impl, can be replaced by
      // call into libraries.
      void _BHGEMM(NDArray A, NDArray B, NDArray C) {
        // A: [Batch, M, K], B: [Batchm K, N], C: [Batch, M, N]
        CHECK_EQ(A->ndim, 3);
        CHECK_EQ(B->ndim, 3);
        CHECK_EQ(C->ndim, 3);
        CHECK_EQ(A->shape[0], B->shape[0]);
        CHECK_EQ(A->shape[2], B->shape[1]);
        int Batch = A->shape[0];
        int M = A->shape[1];
        int K = A->shape[2];
        int N = B->shape[2];
        CHECK_EQ(C->shape[0], Batch);
        CHECK_EQ(C->shape[1], M);
        CHECK_EQ(C->shape[2], N);
        CHECK_EQ(A.DataType(), DataType::Float(16));
        CHECK_EQ(B.DataType(), DataType::Float(16));
        CHECK_EQ(C.DataType(), DataType::Float(16));

        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::GemmBatched<cutlass::half_t,            // ElementA
                                                        cutlass::layout::RowMajor,  // LayoutA
                                                        cutlass::half_t,            // ElementB
                                                        cutlass::layout::RowMajor,  // LayoutB
                                                        cutlass::half_t,            // ElementOutput
                                                        cutlass::layout::RowMajor   // LayoutOutput
                                                        >;
        Gemm gemm_op;
        cutlass::half_t alpha(1.0);
        cutlass::half_t beta(0.0);
        cutlass::layout::ColumnMajor::Stride::Index lda(K);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_A(M * K);
        cutlass::layout::ColumnMajor::Stride::Index ldb(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_B(K * N);
        cutlass::layout::ColumnMajor::Stride::Index ldc(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_C(M * N);
        cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
        cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
        cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::Status status = gemm_op({
            {M, N, K},        //
            {a, lda},         //
            batch_stride_A,   //
            {b, ldb},         //
            batch_stride_B,   //
            {c, ldc},         //
            batch_stride_C,   //
            {c, ldc},         //
            batch_stride_C,   //
            {alpha, beta},    //
            Batch             //
        });
        CHECK(status == cutlass::Status::kSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _BHGEMM); 
"""

GRAPH_PATTERN_CODE_LIST[
    "batch_dense_row_row_row_2/batch_bias_row"
] = """
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm_batched.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {

      using namespace tvm;
      using namespace tvm::runtime;

      // simple specialized impl, can be replaced by
      // call into libraries.
      void _BHGEMM(NDArray A, NDArray B, NDArray Bias, NDArray C) {
        // A: [Batch, M, K], B: [Batch, K, N], Bias: [1, N], C: [Batch, M, N]
        CHECK_EQ(A->ndim, 3);
        CHECK_EQ(B->ndim, 3);
        CHECK_EQ(Bias->ndim, 2);
        CHECK_EQ(C->ndim, 3);
        CHECK_EQ(A->shape[0], B->shape[0]);
        CHECK_EQ(A->shape[2], B->shape[1]);
        CHECK_EQ(Bias->shape[0], 1);
        CHECK_EQ(Bias->shape[1], B->shape[1]);
        int Batch = A->shape[0];
        int M = A->shape[1];
        int K = A->shape[2];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], Batch);
        CHECK_EQ(C->shape[1], M);
        CHECK_EQ(C->shape[2], N);
        CHECK_EQ(A.DataType(), DataType::Float(16));
        CHECK_EQ(B.DataType(), DataType::Float(16));
        CHECK_EQ(C.DataType(), DataType::Float(16));

        // Define the GEMM operation
        using Gemm = cutlass::gemm::device::GemmBatched<
            cutlass::half_t,            // ElementA
            cutlass::layout::RowMajor,  // LayoutA
            cutlass::half_t,            // ElementB
            cutlass::layout::RowMajor,  // LayoutB
            cutlass::half_t,            // ElementOutput
            cutlass::layout::RowMajor,  // LayoutOutput
            cutlass::half_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm75
        >;
        Gemm gemm_op;
        cutlass::half_t alpha(1.0);
        cutlass::half_t beta(0.0);
        cutlass::layout::ColumnMajor::Stride::Index lda(K);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_A(M * K);
        cutlass::layout::ColumnMajor::Stride::Index ldb(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_B(K * N);
        cutlass::layout::ColumnMajor::Stride::Index ld_bias(0);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_bias(0);
        cutlass::layout::ColumnMajor::Stride::Index ldc(N);
        cutlass::layout::ColumnMajor::Stride::Index batch_stride_C(M * N);
        cutlass::half_t* a = reinterpret_cast<cutlass::half_t*>(A->data);
        cutlass::half_t* b = reinterpret_cast<cutlass::half_t*>(B->data);
        cutlass::half_t* c = reinterpret_cast<cutlass::half_t*>(C->data);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);
        cutlass::Status status = gemm_op({
            {M, N, K},           //
            {a, lda},            //
            batch_stride_A,      //
            {b, ldb},            //
            batch_stride_B,      //
            {bias, ld_bias},     //
            batch_stride_bias,   //
            {c, ldc},            //
            batch_stride_C,      //
            {alpha, beta},       //
            Batch                //
        });
        CHECK(status == cutlass::Status::kSuccess);
      }

      }  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({global_symbol}, _BHGEMM); 
"""
