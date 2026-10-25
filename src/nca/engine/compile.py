"""Optional AOT backend that keeps forward ATen operations and compiles backward."""

from functools import partial


def backward_only_backend():
    # This opt-in backend is tested with PyTorch 2.8. Keeping the boxed forward
    # graph avoids Inductor's forward fusion/cast changes on unnormalized Maze.
    from torch._dynamo.backends.common import aot_autograd
    from torch._dynamo.backends.debugging import boxed_nop
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.decomposition import select_decomp_table

    return aot_autograd(
        fw_compiler=boxed_nop,
        bw_compiler=partial(compile_fx_inner, is_backward=True),
        decompositions=select_decomp_table(),
        keep_inference_input_mutations=True,
    )
