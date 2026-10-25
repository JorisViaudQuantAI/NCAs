"""The public-code switches of `NCAConfig` (reasoning-nca, a34414f) and the checkpoint reader of
`nca.public_ckpt`. The parity of the switches with the authors' JAX code, on their published
weights, was checked separately (JAX is not a dependency): see docs/DEVIATIONS.md."""

import io
import pickle

import numpy as np
import pytest
import torch

from nca.embeddings import make_embeddings
from nca.engine.step import StepOptions, make_step
from nca.model import build_model
from nca.public_ckpt import _ArrayUnpickler, read_params

PUBLIC = dict(project_update=True, noise_after_norm=True, norm_init=True, norm_style="norm_eps")


def sudoku_like(**kw):
    return build_model(
        "sudoku_ood", channels=64, c_in=16, c_out=16, heads=4, norm_groups=4, **kw
    ).double()


def test_semantic_embeddings_match_the_public_tables():
    e = make_embeddings(4, 16, "semantic", c_out=2)
    assert e[:, :2].tolist() == [[0, -1], [0, 1], [1, 0], [-1, 0]]
    assert (e[:, 2:] == -1).all()
    e = make_embeddings(5, 32, "semantic", c_out=3)
    assert e[2, :3].tolist() == [1, 0, -1] and (e[:, 3:] == -1).all()
    with pytest.raises(ValueError):
        make_embeddings(10, 32, "semantic", c_out=32)


def test_given_maze_cells_are_fully_confident_with_semantic_embeddings():
    m = build_model("maze_ood", embedding="semantic", output_tokens=(0, 1, 2, 3))
    inputs = torch.tensor([[[2, 3, 0], [0, 2, 3], [2, 0, 0]]])
    given = inputs > 0
    x, _, _ = m.init_state(inputs, given)
    conf = m.confidence(x)[0]
    assert torch.allclose(conf[given[0]], torch.ones(()), atol=1e-3)
    assert (m.decode(x)[0][given[0]] == inputs[0][given[0]]).all()


def test_projection_is_tangent_to_each_group():
    m = sudoku_like(**PUBLIC)
    x = m.normalize(torch.randn(2, 64, 9, 9, dtype=torch.float64))
    d = m.project_tangent(torch.randn_like(x), x)
    dots = (d.view(2, 4, 16, 9, 9) * x.view(2, 4, 16, 9, 9)).sum(2)
    assert dots.abs().max() < 1e-5  # |x| = 1 up to the 1e-6 of the norm


def test_noise_after_norm_leaves_the_noise_unnormalised():
    m = sudoku_like(noise_after_norm=True)
    x, g4, cl = m.init_state(torch.zeros(1, 9, 9, dtype=torch.long), torch.zeros(1, 9, 9).bool())
    noise = torch.zeros_like(x)
    noise[:, 16:, 4, 4] = 0.5  # hidden channels of one cell
    y = m.step(
        x.double(), g4, cl.double(), torch.zeros(1, 1, 9, 9, dtype=torch.float64), noise.double()
    )
    norms = y.view(1, 4, 16, 9, 9).norm(dim=2)
    assert (norms[0, 1:, 4, 4] - 1).abs().max() > 0.1  # the noisy cell is off the sphere
    assert (norms[0, :, 0, 0] - 1).abs().max() < 1e-3  # the others are on it


def test_norm_styles():
    x = torch.full((1, 16, 1, 1), 0.0025, dtype=torch.float64)  # |x| = 0.01
    kw = dict(channels=16, c_in=16, c_out=16, heads=4, norm_groups=1)
    a = build_model("sudoku_ood", **kw).normalize(x)
    b = build_model("sudoku_ood", norm_style="norm_eps", **kw).normalize(x)
    assert abs(a.norm().item() - 0.7071) < 1e-3 and abs(b.norm().item() - 0.9999) < 1e-3


def test_public_init_scales():
    m = build_model("sudoku_ood", init_style="public")
    assert abs(m.update.fc1.weight.std().item() - 1e-3) < 1e-4
    assert abs(m.update.fc2.weight.std().item() - 1e-3) < 1e-4
    assert m.update.fc1.bias.abs().max() == 0
    assert abs(m.perceive.mix.std().item() - 1 / 3) < 0.02
    assert abs(m.perceive.mix.mean().item()) < 0.02


@pytest.mark.parametrize("backend", ["reference", "fused"])
def test_engine_step_honours_the_public_switches(backend):
    torch.manual_seed(3)
    m = sudoku_like(**PUBLIC)
    inputs = torch.randint(1, 10, (2, 9, 9))
    given = torch.rand(2, 9, 9) < 0.3
    x, g4, cl = m.init_state(inputs, given)
    x, cl = x.double(), cl.double()
    fire = (torch.rand(2, 1, 9, 9) < 0.8).double()
    noise = torch.randn_like(x) * 0.1 * (torch.rand(2, 1, 9, 9) < 0.2)
    nmask = (noise != 0).any(1, keepdim=True)
    step = make_step(m, StepOptions.for_backend(backend))
    out = step(x, g4, cl, fire, noise, nmask, None)
    torch.testing.assert_close(out, m.step(x, g4, cl, fire, noise))


def test_reader_rebuilds_arrays_and_refuses_other_objects():
    buf = io.BytesIO()
    pickle.dump({"w": np.arange(6, dtype=np.float32).reshape(2, 3)}, buf)
    buf.seek(0)
    assert _ArrayUnpickler(buf).load()["w"].tolist() == [[0, 1, 2], [3, 4, 5]]

    class Evil:
        def __reduce__(self):
            return (print, ("pwned",))

    with pytest.raises(pickle.UnpicklingError):
        _ArrayUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()


def test_read_params_from_file(tmp_path):
    p = tmp_path / "ema_params.pickle"
    p.write_bytes(pickle.dumps({"update": {"0": {"b1": np.zeros(3, np.float32)}}}))
    assert read_params(p)["update"]["0"]["b1"].shape == (3,)
