import click
import tvm

from tvm.contrib.cutlass.build import handle_conv2d, _get_cutlass_path
from tvm.contrib.cutlass.gen_conv2d import CutlassConv2DProfiler


# "cutlass.conv2d_bias_hardswish": (EpilogueFunctor.LinearCombinationHardSwish, False),
# "cutlass.conv2d_bias_silu": (EpilogueFunctor.LinearCombinationSilu, False),
# "cutlass.conv2d_bias_sigmoid": (EpilogueFunctor.LinearCombinationSigmoid, False),
# "cutlass.conv2d_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
# "cutlass.conv2d_bias": (EpilogueFunctor.LinearCombinationBias, True),
# "cutlass.conv2d": (EpilogueFunctor.LinearCombination, False),
# "cutlass.conv2d_transpose": (EpilogueFunctor.LinearCombination, False),
# "cutlass.conv2d_backward_weight": (EpilogueFunctor.LinearCombination, False),


@click.command()
@click.option("--d", nargs=4, type=int, help="shape of input feature map")
@click.option("--w", nargs=4, type=int, help="shape of weight")
@click.option("--padding", nargs=2, type=int, default=(0, 0), help="padding")
@click.option("--strides", nargs=2, type=int, default=(1, 1), help="strides")
@click.option("--dilation", nargs=2, type=int, default=(1, 1), help="dilation")
@click.option("--sm", default=80, help="SM version")
@click.option("--out_dtype", default="float16", help="Type of A")
@click.option("--data_dtype", default="float16", help="Type of B")
@click.option("--weight_dtype", default="float16", help="Type of C")
@click.option("--op_type", default="cutlass.conv2d", help="Epilogue pattern")
@click.option("--bin_dir", default="./bin", help="Directory to store generated binaries")
def main(d, w, padding, strides, dilation, sm, out_dtype, data_dtype, weight_dtype, op_type, bin_dir):
    cutlass_profiler = CutlassConv2DProfiler(sm, _get_cutlass_path(), bin_dir)
    res = handle_conv2d(
        cutlass_profiler=cutlass_profiler,
        op_type=op_type,
        d_shape=d,
        w_shape=w,
        padding=padding,
        strides=strides,
        dilation=dilation,
        out_dtype=out_dtype,
        data_dtype=data_dtype,
        weight_dtype=weight_dtype,
        use_3xtf32=False,
        split_k_slices=[1],
        profile_all_alignments=True,
        find_first_valid=False,
        use_multiprocessing=False,
    )
    print(res["cutlass_op_def"])
    print(res["cutlass_op_name"])


if __name__ == "__main__":
    main()
