"""Evaluations of scientific runs load official data only, whatever config the checkpoint embeds."""

import pytest

from nca.cli import eval_source
from nca.tasks import get_task


def _cfg(train=None, test=None):
    task = {"name": "maze_ood", "train": train or {"size": 9}}
    if test is not None:
        task["test"] = test
    return {"task": task}


def test_old_checkpoint_without_source_is_evaluated_on_official_data():
    # a final.pt whose embedded config predates `source: official` and has no task.test
    assert eval_source(_cfg(), {}) == "official"
    assert eval_source(_cfg(), {"data": {"train_source": "official"}}) == "official"
    assert eval_source(_cfg(train={"size": 9, "source": "official"}), {}) == "official"


def test_smoke_and_generated_runs_keep_their_kind_of_data():
    assert eval_source(_cfg(train={"size": 9, "source": "auto"}), {}) is None
    assert eval_source(_cfg(), {"data": {"train_source": "generated"}}) == "generated"


def test_synthetic_arc_runs_stay_synthetic():
    # review of 2026-10-13: a synthetic ARC run must not be evaluated on official tasks
    assert eval_source(_cfg(train={"source": "synthetic"}), {}) == "synthetic"
    assert eval_source(_cfg(), {"data": {"train_source": "synthetic"}}) == "synthetic"


def test_explicit_test_source_wins():
    assert eval_source(_cfg(test={"source": "generated"}), {}) == "generated"


def test_official_source_never_falls_back_to_the_generator(tmp_path):
    # with the source decided above, missing official data is an error, not generated boards
    task = get_task("maze_ood")
    with pytest.raises(FileNotFoundError):
        task.load(
            "test", str(tmp_path), size=9, n=4, download=False, source=eval_source(_cfg(), {})
        )
