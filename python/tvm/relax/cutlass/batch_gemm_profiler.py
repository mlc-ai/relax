import click
import tvm

from tvm.contrib.cutlass.build import select_gemm_kernel, _get_cutlass_path
from tvm.contrib.cutlass.gen_gemm import CutlassGemmProfiler


# "cutlass.batch_matmul": (EpilogueFunctor.LinearCombination, False),


@click.command()
@click.option("--b", default=2, help="bacth dimension")
@click.option("--m", default=1024, help="M dimension")
@click.option("--n", default=1024, help="N dimension")
@click.option("--k", default=1024, help="K dimension")
@click.option("--sm", default=80, help="SM version")
@click.option("--typea", default="float16", help="Type of A")
@click.option("--typeb", default="float16", help="Type of B")
@click.option("--typec", default="float16", help="Type of C")
@click.option("--layouta", default="row", help="Layout of A")
@click.option("--layoutb", default="row", help="Layout of B")
@click.option("--layoutc", default="row", help="Layout of C")
@click.option("--op_type", default="cutlass.dense", help="Epilogue pattern")
@click.option("--bin_dir", default="./bin", help="Directory to store generated binaries")
def main(b, m, n, k, sm, typea, typeb, typec, layouta, layoutb, layoutc, op_type, bin_dir):
    cutlass_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), bin_dir)
    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        m,
        k,
        n,
        typec,
        typea,
        typeb,
        layoutc,
        layouta,
        layoutb,
        False,
        True,
        False,
        True,
    )
    print(name)
    print(cutlass_op_def)


if __name__ == "__main__":
    main()
