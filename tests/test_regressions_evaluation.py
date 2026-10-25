"""Regression tests for evaluation fixes found in code review (2026-10-13)."""

import numpy as np
import torch

from nca.experiments import adaptive_compute, tts_curves
from nca.model import NCAConfig, build_model
from nca.rollout import RolloutConfig, conf_at_k, rollout
from nca.tasks import get_task
from nca.tasks.base import Boards
from nca.tasks.sudoku import EXTREME_BINS, _stratify, cache_path


def _sudoku_state(m, B=2):
    inputs = torch.randint(0, 10, (B, 9, 9), generator=torch.Generator().manual_seed(0))
    return m.init_state(inputs, inputs > 0)


# 3. a cell that does not fire stays frozen, including the normalisation ------------------


def test_non_firing_cells_stay_frozen_from_the_gaussian_initial_state():
    m = build_model("sudoku_extreme")  # norm_fired_only=True
    torch.nn.init.normal_(m.update.fc2.weight, std=0.1)
    x, g4, clamp = _sudoku_state(m)  # not pre-normalised: the case the old test missed
    x1 = m.step(x, g4, clamp, torch.zeros(2, 1, 9, 9))
    assert torch.equal(x1, x)
    # cells that fire are normalised, the others are untouched
    fm = torch.zeros(2, 1, 9, 9)
    fm[:, :, :4] = 1
    x2 = m.step(x, g4, clamp, fm)
    free = ~g4.expand_as(x)
    rows = torch.zeros_like(free)
    rows[:, :, 4:] = True
    assert torch.equal(x2[free & rows], x[free & rows])
    assert not torch.allclose(x2[free & ~rows], x[free & ~rows])


def test_old_checkpoints_keep_the_historical_normalisation():
    cfg = build_model("sudoku_ood").cfg.to_dict()
    cfg.pop("norm_fired_only")  # a model_config saved before 2026-10-13
    m = build_model("sudoku_ood")
    m.cfg = NCAConfig.from_dict(cfg)
    assert m.cfg.norm_fired_only is False
    x, g4, clamp = _sudoku_state(m)
    x1 = m.step(x, g4, clamp, torch.zeros(2, 1, 9, 9))
    assert not torch.equal(x1, x)  # every cell renormalised, as the published runs were trained


# 4. Sudoku-Extreme: benchmark-weighted global score, stratify in the cache key -----------


def test_stratified_sample_carries_benchmark_weights():
    rng = np.random.default_rng(0)
    N = 7000
    # an unbalanced benchmark: most boards in the easiest bin
    rating = np.concatenate([np.zeros(6000), rng.integers(1, 100, N - 6000)]).astype(np.int64)
    b = Boards(np.zeros((N, 9, 9), int), np.zeros((N, 9, 9), int), np.zeros((N, 9, 9), bool))
    b.meta["rating"] = rating
    s = _stratify(b, "rating", 70, np.random.default_rng(1))
    w = s.meta["weight"]
    assert np.isclose(w.sum(), 1.0)
    share = np.bincount(s.meta["bin"], weights=w, minlength=len(EXTREME_BINS))
    full = np.bincount(b.meta["bin"], minlength=len(EXTREME_BINS)) / N
    assert np.allclose(share, full)


def test_weighted_conf_at_k():
    metric = np.array([[1.0], [0.0]])
    conf = np.ones((2, 1))
    alive = np.ones((2, 1), bool)
    assert conf_at_k(metric, conf, alive, [1])[1] == 0.5
    assert conf_at_k(metric, conf, alive, [1], weights=np.array([0.9, 0.1]))[1] == 0.9


def test_stratify_is_part_of_the_cache_identity():
    a = cache_path("d", "extreme", "test", 100, 0, "official", "strat_v2")
    b = cache_path("d", "extreme", "test", 100, 0, "official", "uniform")
    assert a != b


# 5. adaptive results are reused only for the same experiment ----------------------------


def _maze(tmp_path, n=3):
    task = get_task("maze_ood")
    return task, task.load("test", str(tmp_path), size=9, n=n, download=False, source="generated")


def test_adaptive_reuse_needs_the_same_fingerprint(tmp_path):
    task, boards = _maze(tmp_path)
    m = build_model("maze_ood").eval()
    kw = dict(record_every=4, variants=("uniform",))
    first = adaptive_compute(m, task, boards, 8, 0.8, 0.4, 0.9, seed=0, **kw)
    same = adaptive_compute(m, task, boards, 8, 0.8, 0.4, 0.9, seed=0, previous=first, **kw)
    assert same["fingerprint"] == first["fingerprint"]
    assert same["variants"]["uniform"] is first["variants"]["uniform"]  # reused
    for change in (dict(seed=1), dict(p_fire=0.5, seed=0), dict(tau=0.5, seed=0)):
        args = dict(p_fire=0.8, p_low=0.4, tau=0.9) | {
            k: v for k, v in change.items() if k != "seed"
        }
        other = adaptive_compute(
            m, task, boards, 8, args["p_fire"], args["p_low"], args["tau"],
            seed=change.get("seed", 0), previous=first, **kw,
        )  # fmt: skip
        assert other["fingerprint"] != first["fingerprint"]
        assert other["variants"]["uniform"] is not first["variants"]["uniform"]  # recomputed


# 7. --amp reaches the rollout and is recorded --------------------------------------------


def test_amp_is_passed_and_recorded(tmp_path, monkeypatch):
    import nca.experiments as ex

    seen = []
    real = ex.rollout

    def spy(model, task, boards, rc, **kw):
        seen.append(rc.amp)
        return real(model, task, boards, rc, **kw)

    monkeypatch.setattr(ex, "rollout", spy)
    task, boards = _maze(tmp_path)
    res = tts_curves(build_model("maze_ood").eval(), task, boards, 4, 1, 2, amp=True)
    assert seen == [True] and res["amp"] is True


# 8. rollouts shorter than the recording interval ------------------------------------------


def test_short_rollout_records_its_last_step(tmp_path):
    task, boards = _maze(tmp_path)
    res = rollout(
        build_model("maze_ood").eval(), task, boards, RolloutConfig(depth=4, k=1, record_every=8)
    )
    assert res.steps.tolist() == [4]


# 10. Sobel filters per channel -------------------------------------------------------------


def test_sobel_filters_are_identity_sx_sy_for_every_channel():
    from nca.model import SobelPerception

    p = SobelPerception(3)
    w = p.weight.view(3, 3, 3, 3)  # (channel, filter, 3, 3)
    for c in range(3):
        assert w[c, 0, 1, 1] == 1 and w[c, 0].abs().sum() == 1  # identity
        assert torch.allclose(w[c, 2], w[c, 1].T)  # sy = sx transposed
        assert not torch.allclose(w[c, 1], w[c, 0])


# Third review of 2026-10-13 (89470e2) -------------------------------------------------------


def test_fingerprint_covers_record_every_model_config_and_given(tmp_path):
    from nca.experiments import experiment_fingerprint

    task, boards = _maze(tmp_path)
    m = build_model("maze_ood").eval()
    base = experiment_fingerprint(m, boards, record_every=4)
    assert experiment_fingerprint(m, boards, record_every=1) != base
    m2 = build_model("maze_ood", sigma_init=0.3).eval()
    m2.load_state_dict(m.state_dict())
    assert experiment_fingerprint(m2, boards, record_every=4) != base
    b2 = Boards(boards.inputs, boards.targets, ~boards.given, dict(boards.meta))
    assert experiment_fingerprint(m, b2, record_every=4) != base


def test_adaptive_recomputes_when_record_every_changes(tmp_path):
    task, boards = _maze(tmp_path)
    m = build_model("maze_ood").eval()
    first = adaptive_compute(m, task, boards, 8, 0.8, 0.4, 0.9, 4, variants=("uniform",))
    other = adaptive_compute(
        m, task, boards, 8, 0.8, 0.4, 0.9, 1, variants=("uniform",), previous=first
    )
    assert other["variants"]["uniform"]["steps"] == list(range(1, 9))


def test_resume_restores_the_historical_model_config(tmp_path):
    import json

    import pytest

    from nca.train import restore_model_config

    m, ema = build_model("sudoku_extreme"), build_model("sudoku_extreme")
    stored = m.cfg.to_dict()
    stored.pop("norm_fired_only")  # the run started before the option existed
    (tmp_path / "run.json").write_text(json.dumps({"model_config": stored}))
    assert m.cfg.norm_fired_only is True
    restore_model_config(m, ema, tmp_path / "run.json")
    assert m.cfg.norm_fired_only is False and ema.cfg.norm_fired_only is False
    stored["channels"] = stored["channels"] + 1  # a preset that really changed under the run
    (tmp_path / "run.json").write_text(json.dumps({"model_config": stored}))
    with pytest.raises(RuntimeError, match="channels"):
        restore_model_config(
            build_model("sudoku_extreme"), build_model("sudoku_extreme"), tmp_path / "run.json"
        )


def test_global_curves_are_weighted_on_a_stratified_sample(tmp_path):
    task, boards = _maze(tmp_path, n=4)
    boards.meta["weight"] = np.array([0.7, 0.1, 0.1, 0.1])
    res = tts_curves(build_model("maze_ood").eval(), task, boards, 4, 2, 2)
    assert res["weighted"] is True
    for key in ("conf_at_k_weighted", "pass_at_k_weighted", "depth_curve_weighted"):
        assert key in res
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "make_figures", root / "scripts" / "make_figures.py"
    )
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    assert mf.overall(res, "depth_curve") == res["depth_curve_weighted"]


def test_aggregate_does_not_label_old_stratified_scores_as_weighted():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("aggregate", root / "scripts" / "aggregate.py")
    ag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ag)
    key = "conf_at_k_weighted.1|conf_at_k.1@weighted=false"
    old = {"conf_at_k": {"1": 0.571}}  # stratified, written before the weights existed
    assert ag.dig(old, key) is None
    uniform = {"conf_at_k": {"1": 0.94}, "weighted": False}
    assert ag.dig(uniform, key) == 0.94
    new = {"conf_at_k": {"1": 0.571}, "conf_at_k_weighted": {"1": 0.94}, "weighted": True}
    assert ag.dig(new, key) == 0.94
