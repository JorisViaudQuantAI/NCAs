"""Load the checkpoints published with the authors' code into this implementation.

The authors' repository (github.com/paradigms-of-intelligence/reasoning-nca, commit a34414f)
ships, for each task, `checkpoints/<Task>/config.json` (the notebook's config) and
`ema_params.pickle` (the EMA weights as a nested dict of JAX arrays). `load_public` turns them
into an `NCA` whose config switches reproduce the public code's equations (semantic maze
embeddings, readout over the whole vocabulary, update projection, noise after normalisation,
`x / (|x| + 1e-6)`), so the weights run unchanged. Evaluating these models with our scripts is a
positive control: it measures what the authors' training reaches under our evaluation, and
checks our evaluation against the paper's numbers.

The pickle is read with a restricted unpickler that only rebuilds numpy arrays (JAX arrays are
rebuilt as numpy arrays), so neither JAX nor arbitrary code is needed to read it.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .model import NCA, NCAConfig


def _reconstruct_array(fun, args, arr_state, aval_state):
    """Stand-in for `jax._src.array._reconstruct_array`: the numpy value of the array."""
    value = fun(*args)
    value.__setstate__(arr_state)
    return value


class _ArrayUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) == ("jax._src.array", "_reconstruct_array"):
            return _reconstruct_array
        if (module, name) in self.ALLOWED:
            mod = np._core.multiarray if name == "_reconstruct" else np
            return getattr(mod, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def read_params(path: str | Path) -> dict:
    """The nested parameter dict of `ema_params.pickle`, with numpy arrays as leaves."""
    with open(path, "rb") as f:
        return _ArrayUnpickler(f).load()


def public_model_config(pc: dict) -> NCAConfig:
    """`NCAConfig` reproducing the public code's model for its config dict `pc`."""
    if pc["NUM_LAYERS"] != 1 or pc["UPDATE_TYPE"] != "mlp" or pc["CONNECTIVITY"] != "local":
        raise ValueError("only the single-layer local MLP models are supported")
    if pc.get("SENSING_NORM", "none") != "none" or pc.get("STEP_SIZE", 1.0) != 1.0:
        raise ValueError("SENSING_NORM / STEP_SIZE differ from the published checkpoints")
    maze = pc["DSET_NAME"].startswith("m")
    group_norm = pc["NORMALIZATION"] == "group_norm"
    if pc["NORMALIZATION"] not in ("none", "group_norm"):
        raise ValueError(f"unsupported normalisation {pc['NORMALIZATION']!r}")
    perception = {"convolution": "kernels", "fixed_attention": "fixed_attn"}[pc["SENSING_TYPE"]]
    padding = {"wrap": "wrap", "token_emb": "wall"}[pc["SENSING_PADDING"]]
    if padding == "wall" and pc["INPUT_EMB_D"] != pc["CHANNEL_N"]:
        raise ValueError("token_emb padding needs INPUT_EMB_D == CHANNEL_N")
    return NCAConfig(
        vocab_size=pc["VOCAB_SIZE"],
        channels=pc["CHANNEL_N"],
        c_in=pc["INPUT_EMB_D"],
        c_out=pc["OUTPUT_EMB_D"],
        output_tokens=tuple(range(pc["VOCAB_SIZE"])),
        embedding="semantic" if maze else "orthonormal",
        perception=perception,
        heads=pc["SENSING_HEADS"],
        grid_size=(pc["H"], pc["W"]) if perception == "fixed_attn" else None,
        padding=padding,
        wall_token=pc["PADDING_TOKEN_ID"],
        update="mlp",
        expansion=pc["UPDATE_EXPANSION"],
        hidden_from_perception=perception == "kernels",
        norm_groups=pc["NUM_UNITS"] if group_norm else 0,
        norm_fired_only=False,
        sigma_init=pc["INIT_NOISE_STD"],
        fire_rate=pc["CELL_FIRE_RATE"],
        head={"l2_similarity": "l2", "cosine_similarity": "cosine"}[pc["PRED_TYPE"]],
        zero_init_update=False,
        project_update=group_norm,
        noise_after_norm=True,
        norm_init=True,
        norm_style="norm_eps",
        init_style="public",
    )


def load_public(ckpt_dir: str | Path) -> tuple[NCA, dict]:
    """(model, public config dict) for `checkpoints/<Task>` of the authors' repository."""
    d = Path(ckpt_dir)
    pc = json.loads((d / "config.json").read_text())
    if pc.get("CONF_TYPE") != "max":
        raise ValueError("only CONF_TYPE 'max' (no learned confidence head) is supported")
    model = NCA(public_model_config(pc))
    p = read_params(d / "ema_params.pickle")
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32).copy())  # noqa: E731
    cfg = model.cfg
    with torch.no_grad():
        emb = t(p["input_embed"])
        if emb.shape != model.embed.shape:
            raise ValueError(f"input_embed {tuple(emb.shape)} != {tuple(model.embed.shape)}")
        if cfg.embedding == "semantic" and not torch.equal(emb, model.embed):
            raise ValueError("the checkpoint's maze embeddings differ from `semantic`")
        model.embed.copy_(emb)
        lo = cfg.out_offset - cfg.in_offset
        model.out_embed.copy_(emb[list(cfg.output_tokens)][:, lo : lo + cfg.c_out])
        model.pad_vec.zero_()
        if cfg.padding == "wall":
            model.pad_vec[cfg.in_slice] = emb[cfg.wall_token]
        s = p["sensing"]["0"]
        if cfg.perception == "kernels":
            k = t(s["kernels"])  # (heads, 9), neighbours row-major as our OFFSETS
            model.perceive.weight.copy_(k.view(cfg.heads, 1, 3, 3))
        else:
            fa = t(s["fixed_attention"])  # (H, W, heads, 1, 9)
            model.perceive.mix.copy_(fa[:, :, :, 0, :].permute(2, 3, 0, 1))
            v = t(s["v"])  # (heads, C / heads, C)
            model.perceive.value.weight.copy_(
                v.reshape(cfg.channels, cfg.channels)[..., None, None]
            )
        u = p["update"]["0"]
        model.update.fc1.weight.copy_(t(u["w1"])[..., None, None])
        model.update.fc1.bias.copy_(t(u["b1"]))
        model.update.fc2.weight.copy_(t(u["w2"])[..., None, None])
        model.update.fc2.bias.copy_(t(u["b2"]))
    return model.eval(), pc
