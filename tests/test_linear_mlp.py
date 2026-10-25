"""Linear and 1x1 convolution dispatch must share weights, values and derivatives."""

import pytest
import torch

from nca.engine.ops import linear_cell_update
from nca.model import MLPUpdate, SwiGLUUpdate


@pytest.mark.parametrize("kind,module", [("mlp", MLPUpdate), ("swiglu", SwiGLUUpdate)])
@pytest.mark.parametrize("channels_last", [False, True])
def test_linear_cell_update_preserves_outputs_and_all_gradients(kind, module, channels_last):
    torch.manual_seed(76)
    z = torch.randn(2, 13, 5, 7, dtype=torch.float64)
    if channels_last:
        z = z.contiguous(memory_format=torch.channels_last)
    z.requires_grad_()
    update = module(13, 11, 17, False).double()
    ref = update(z)
    out = linear_cell_update(z, update, kind)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)
    upstream = torch.randn_like(ref)
    params = (z, *update.parameters())
    a = torch.autograd.grad(ref, params, upstream, retain_graph=True)
    b = torch.autograd.grad(out, params, upstream)
    for u, v in zip(a, b, strict=True):
        torch.testing.assert_close(u, v, atol=1e-11, rtol=1e-11)
