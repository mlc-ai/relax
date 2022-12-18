import tvm
from tvm import tir
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.relax.transform import OperatorLegalizer
from tvm import register_func

OP_PATTERN_LIST = list()
OP_PATTERN_FUNC_LIST = dict()
OP_PATTERN_VARS_LIST = dict()
GRAPH_PATTERN_CODE_LIST = dict()


def register_pattern():
    def register(func):
        func()
        return func

    return register


def get_value(evaluated_symbols, pattern_name):
    return [evaluated_symbols[symbol] for symbol in OP_PATTERN_VARS_LIST[pattern_name]]


def get_graph_pattern_code(cutlass_op, *args, **kwargs):
    cutlass_op = [str(st) for st in cutlass_op]
    pattern = "/".join(cutlass_op)
    if pattern not in GRAPH_PATTERN_CODE_LIST:
        raise tvm.TVMError("Cannot find graph pattern code for cutlass op: {}".format(cutlass_op))
    codegen = GRAPH_PATTERN_CODE_LIST["/".join(cutlass_op)]
    if isinstance(codegen, str):
        return codegen
    elif callable(codegen):
        return codegen(*args, **kwargs)


@register_func("tvm.relax.cutlass.op_pattern_stitch")
def op_pattern_stitch(evaluated_symbols, evaluated_buffers, matched_pattern_names):
    print("matched_pattern_names", matched_pattern_names)
    print("evaluated_symbols", evaluated_symbols)
    print("evaluated_buffers", evaluated_buffers)
    if len(matched_pattern_names) >= 3:
        assert len(evaluated_symbols) >= 3
        assert len(evaluated_buffers) >= 3
        if (
            matched_pattern_names[0] == "dense_row_row_row"
            and matched_pattern_names[1] == "bias_row"
            and matched_pattern_names[2] == "relu"
        ):
            # dense_row_row_row + bias_row + relu
            m_dense, n_dense, k_dense = get_value(evaluated_symbols[0], "dense_row_row_row")
            m_bias, n_bias = get_value(evaluated_symbols[1], "bias_row")
            m_relu, n_relu = get_value(evaluated_symbols[2], "relu")
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
                return [get_graph_pattern_code(matched_pattern_names[:3]), 3]
    if len(matched_pattern_names) >= 2:
        assert len(evaluated_symbols) >= 2
        assert len(evaluated_buffers) >= 2
        # dense_row_row_row + bias_row
        if (
            matched_pattern_names[0] == "dense_row_row_row"
            and matched_pattern_names[1] == "bias_row"
        ):
            m_dense, n_dense, k_dense = get_value(evaluated_symbols[0], "dense_row_row_row")
            m_bias, n_bias = get_value(evaluated_symbols[1], "bias_row")
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                return [get_graph_pattern_code(matched_pattern_names[:2]), 2]
        # batch_dense_row_row_row + batch_bias_row
        if (
            matched_pattern_names[0] == "batch_dense_row_row_row"
            and matched_pattern_names[1] == "batch_bias_row"
        ):
            b_dense, m_dense, n_dense, k_dense = get_value(
                evaluated_symbols[0], "batch_dense_row_row_row"
            )
            b_bias, m_bias, n_bias = get_value(evaluated_symbols[1], "batch_bias_row")
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if b_dense == b_bias and m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                return [get_graph_pattern_code(matched_pattern_names[:2]), 2]
        # dense_row_row_row + bias_row
        if (
            matched_pattern_names[0] == "batch_dense_row_row_row_2"
            and matched_pattern_names[1] == "batch_bias_row"
        ):
            b_dense, m_dense, n_dense, k_dense = get_value(
                evaluated_symbols[0], "batch_dense_row_row_row_2"
            )
            b_bias, m_bias, n_bias = get_value(evaluated_symbols[1], "batch_bias_row")
            A_dense, B_dense, C_dense = evaluated_buffers[0]
            A_bias, B_bias, C_bias = evaluated_buffers[1]
            if b_dense == b_bias and m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
                return [get_graph_pattern_code(matched_pattern_names[:2]), 2]
        # padding2d_NHWC + conv2d_NHWC
        if (
            matched_pattern_names[0] == "padding_2d_NHWC"
            and matched_pattern_names[1] == "conv2d_NHWC"
        ):
            (
                N_pad,
                H_pad,
                W_pad,
                C_pad,
                pH_pad,
                pW_pad,
                lH_pad,
                lW_pad,
                rH_pad,
                rW_pad,
            ) = get_value(evaluated_symbols[0], "padding_2d_NHWC")
            (
                N_conv,
                pH_conv,
                pW_conv,
                H_conv,
                W_conv,
                C_conv,
                O_conv,
                KH_conv,
                KW_conv,
                stride_h_conv,
                stride_w_conv,
                dilation_h_conv,
                dilation_w_conv,
            ) = get_value(evaluated_symbols[1], "conv2d_NHWC")
            A, A_pad = evaluated_buffers[0]
            A_pad_conv, B_conv, out_conv = evaluated_buffers[1]
            if (
                N_pad == N_conv
                and pH_pad == pH_conv
                and pW_pad == pW_conv
                and C_pad == C_conv
                and A_pad == A_pad_conv
            ):
                if (
                    lH_pad == pH_pad - rH_pad
                    and lW_pad == pW_pad - rW_pad
                    and lH_pad + H_pad == rH_pad
                    and lW_pad + W_pad == rW_pad
                ):
                    D_shape = (N_pad, H_pad, W_pad, C_pad)
                    W_shape = (O_conv, KH_conv, KW_conv, C_conv)
                    padding = (lH_pad, lW_pad)
                    strides = (stride_h_conv, stride_w_conv)
                    dilation = (dilation_h_conv, dilation_w_conv)
                    return [
                        get_graph_pattern_code(
                            matched_pattern_names[:2],
                            d=D_shape,
                            w=W_shape,
                            padding=padding,
                            strides=strides,
                            dilation=dilation,
                        ),
                        2,
                    ]
    if len(matched_pattern_names) >= 1:
        assert len(evaluated_symbols) >= 1
        assert len(evaluated_buffers) >= 1
        if matched_pattern_names[0] == "dense_row_row_row":
            # dense_row_row_row
            return [get_graph_pattern_code(matched_pattern_names[:1]), 1]
        elif matched_pattern_names[0] == "batch_dense_row_row_row":
            # batch_dense_row_row_row
            return [get_graph_pattern_code(matched_pattern_names[:1]), 1]
        elif matched_pattern_names[0] == "batch_dense_row_row_row_2":
            # batch_dense_row_row_row_2
            return [get_graph_pattern_code(matched_pattern_names[:1]), 1]
    return ["", 0]


A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"


@register_func("tvm.relax.cutlass.get_op_pattern_list")
def get_op_pattern_list():
    return OP_PATTERN_LIST


@register_func("tvm.relax.cutlass.get_op_pattern_func")
def get_op_pattern_func(name):
    return OP_PATTERN_FUNC_LIST[name]


@register_pattern()
def dense_row_row_row():
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    K = tir.Var("K", "int64")

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                C = R.nn.matmul(A, B, out_dtype=C_TYPE)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "dense_row_row_row"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["matmul"]
    OP_PATTERN_VARS_LIST[name] = [M, N, K]


@register_pattern()
def bias_row():
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((M, N), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((1, N), B_TYPE))  # pylint: disable=invalid-name
                C = R.add(A, B)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "bias_row"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["add"]
    OP_PATTERN_VARS_LIST[name] = [M, N]


@register_pattern()
def batch_bias_row():
    batch = tir.Var("batch", "int64")
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((batch, M, N), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((1, N), B_TYPE))  # pylint: disable=invalid-name
                C = R.add(A, B)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "batch_bias_row"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["add"]
    OP_PATTERN_VARS_LIST[name] = [batch, M, N]


@register_pattern()
def relu():
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((M, N), A_TYPE))  # pylint: disable=invalid-name
                B = R.nn.relu(A)
                R.func_ret_value(B)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "relu"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["relu"]
    OP_PATTERN_VARS_LIST[name] = [M, N]


@register_pattern()
def batch_dense_row_row_row():
    batch = tir.Var("batch", "int64")
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    K = tir.Var("K", "int64")

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((batch, M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((K, N), B_TYPE))  # pylint: disable=invalid-name
                C = R.nn.matmul(A, B, out_dtype=C_TYPE)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "batch_dense_row_row_row"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["matmul"]
    OP_PATTERN_VARS_LIST[name] = [batch, M, N, K]


@register_pattern()
def batch_dense_row_row_row_2():
    batch = tir.Var("batch", "int64")
    M = tir.Var("M", "int64")
    N = tir.Var("N", "int64")
    K = tir.Var("K", "int64")

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg("A", R.tensor((batch, M, K), A_TYPE))  # pylint: disable=invalid-name
                B = R.arg("B", R.tensor((batch, K, N), B_TYPE))  # pylint: disable=invalid-name
                C = R.nn.matmul(A, B, out_dtype=C_TYPE)
                R.func_ret_value(C)
    relax_mod = ib.get()
    relax_mod = OperatorLegalizer(relax_mod).transform()
    name = "batch_dense_row_row_row_2"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = relax_mod["matmul"]
    OP_PATTERN_VARS_LIST[name] = [batch, M, N, K]


@register_pattern()
def copy_4d():
    N = tir.Var("N", "int64")
    H = tir.Var("H", "int64")
    W = tir.Var("W", "int64")
    C = tir.Var("C", "int64")

    from tvm.script import tir as T

    @tvm.script.ir_module
    class Copy4D:
        @T.prim_func
        def main(A: T.Buffer((N, H, W, C), A_TYPE), B: T.Buffer((N, H, W, C), B_TYPE)) -> None:
            for n, h, w, c in T.grid(N, H, W, C):
                with T.block("copy"):
                    vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
                    T.reads([A[vn, vh, vw, vc]])
                    T.writes([B[vn, vh, vw, vc]])
                    B[vn, vh, vw, vc] = A[vn, vh, vw, vc]

    mod = Copy4D
    name = "copy_4d"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = mod["main"]
    OP_PATTERN_VARS_LIST[name] = [N, H, W, C]


@register_pattern()
def padding_2d():
    N = tir.Var("N", "int64")
    H = tir.Var("H", "int64")
    W = tir.Var("W", "int64")
    pH = tir.Var("pH", "int64")
    pW = tir.Var("pW", "int64")
    lH = tir.Var("lH", "int64")
    lW = tir.Var("lW", "int64")
    rH = tir.Var("rH", "int64")
    rW = tir.Var("rW", "int64")
    C = tir.Var("C", "int64")

    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                A = T.arg("A", T.buffer_decl((N, H, W, C), A_TYPE))
                B = T.arg("B", T.buffer_decl((N, pH, pW, C), B_TYPE))
                T.func_name("main")
                with T.grid(N, pH, pW, C) as (n, ph, pw, c):
                    with T.block("copy"):
                        vn, vph, vpw, vc = T.axis.remap("SSSS", [n, ph, pw, c])
                        T.reads([A[vn, vph - lH, vpw - lW, vc]])
                        T.writes([B[vn, vph, vpw, vc]])
                        T.buffer_store(
                            B,
                            T.if_then_else(
                                tvm.tir.all(lH <= vph, vph < rH, lW <= vpw, vpw < rW),
                                A[vn, vph - lH, vpw - lW, vc],
                                T.float16(0.0),
                            ),
                            [vn, vph, vpw, vc],
                        )
    mod = ib.get()
    name = "padding_2d_NHWC"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = mod["main"]
    OP_PATTERN_VARS_LIST[name] = [N, H, W, C, pH, pW, lH, lW, rH, rW]


@register_pattern()
def conv2d():
    N = tir.Var("N", "int64")
    pH = tir.Var("pH", "int64")
    pW = tir.Var("pW", "int64")
    H = tir.Var("H", "int64")
    W = tir.Var("W", "int64")
    C = tir.Var("C", "int64")
    O = tir.Var("K", "int64")
    KH = tir.Var("R", "int64")
    KW = tir.Var("S", "int64")
    StrideH = tir.Var("StrideH", "int64")
    StrideW = tir.Var("StrideW", "int64")
    DilateH = tir.Var("DilateH", "int64")
    DilateW = tir.Var("DilateW", "int64")

    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:
        with I.ir_module() as frame:
            with T.prim_func():
                A = T.arg("A", T.buffer_decl((N, pH, pW, C), A_TYPE))
                B = T.arg("B", T.buffer_decl((O, KH, KW, C), B_TYPE))
                out = T.arg("out", T.buffer_decl((N, H, W, O), C_TYPE))
                T.func_name("main")
                with T.grid(N, H, W, O, KH, KW, C) as (n, h, w, o, rh, rw, c):
                    with T.block("conv"):
                        vn, vh, vw, vo, vrh, vrw, vc = T.axis.remap(
                            "SSSSRRR", [n, h, w, o, rh, rw, c]
                        )
                        T.reads(
                            [
                                A[
                                    vn,
                                    vrh * DilateH + vh * StrideH,
                                    vrw * DilateW + vw * StrideW,
                                    vc,
                                ],
                                B[vo, vrh, vrw, vc],
                            ]
                        )
                        T.writes([out[vn, vh, vw, vo]])
                        with T.init():
                            T.buffer_store(out, T.float16(0.0), [vn, vh, vw, vo])
                        T.buffer_store(
                            out,
                            out[vn, vh, vw, vo]
                            + A[vn, vrh * DilateH + vh * StrideH, vrw * DilateW + vw * StrideW, vc]
                            * B[vo, vrh, vrw, vc],
                            [vn, vh, vw, vo],
                        )
    mod = ib.get()
    name = "conv2d_NHWC"
    OP_PATTERN_LIST.append(name)
    OP_PATTERN_FUNC_LIST[name] = mod["main"]
    OP_PATTERN_VARS_LIST[name] = [N, pH, pW, H, W, C, O, KH, KW, StrideH, StrideW, DilateH, DilateW]


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


def conv2d_NHWC_codegen(d, w, padding, strides, dilation):
    print(d)
    print(w)
    print(padding)
    print(strides)
    print(dilation)

    """
    // Conv2dFprop Optimized kernel instance "cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align1"
    using cutlass_tensorop_h1688fprop_optimized_128x64_32x2_nhwc_align1 =
        typename cutlass::conv::kernel::DefaultConv2dFprop<
            cutlass::half_t,
            cutlass::layout::TensorNHWC,
            cutlass::half_t,
            cutlass::layout::TensorNHWC,
            cutlass::half_t,
            cutlass::layout::TensorNHWC,
            cutlass::half_t,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32 >,
            cutlass::gemm::GemmShape<16, 8, 8>,
            
            cutlass::epilogue::thread::LinearCombination<
                cutlass::half_t,
                1,
                cutlass::half_t,
                cutlass::half_t
            >,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
            2,
            cutlass::arch::OpMultiplyAdd,
            cutlass::conv::IteratorAlgorithm::kOptimized,
            cutlass::conv::StrideSupport::kStrided,
            1,
            1
            >::Kernel;
    """

    src = """
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
    return "123123"


GRAPH_PATTERN_CODE_LIST["padding_2d_NHWC/conv2d_NHWC"] = conv2d_NHWC_codegen
