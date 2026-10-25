"""Reference backend vs a preset's own `engine:` section, same protocol as scripts/check_engine.py.

    python check_preset.py --config configs/budget_optimized/sudoku_ood.yaml --batch 32 --out ...
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from nca.engine.bench import (  # noqa: E402
    DEFAULT_TOL,
    compare_backends,
    hardware_info,
    iteration_seeds,
    make_batch,
    within_tolerance,
)
from nca.engine.engine import EngineConfig  # noqa: E402
from nca.engine.train import loss_weights, resolve_train_cfg  # noqa: E402
from nca.train import build_run_model, build_task_and_data  # noqa: E402
from nca.utils import get_device, load_config, save_json, seed_everything  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--steps", default=None)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dev = get_device("auto")
cfg = load_config(args.config, args.set)
tr = resolve_train_cfg(cfg)
tr["batch_size"] = args.batch
task, boards, _ = build_task_and_data(cfg["task"], cfg.get("data_root", "data"))
seed_everything(args.seed)
model = build_run_model(cfg).to(dev)
batch = make_batch(model, task, boards, tr["batch_size"], args.seed)
preset = EngineConfig.from_dict(cfg["engine"])
preset_d = {**dataclasses.asdict(preset), "checkpoint_segment": None}
steps = [int(s) for s in (args.steps or f"1,8,{tr['chunk']}").split(",")]
report = dict(config=args.config, overrides=args.set, engine=preset_d, train=tr,
              hardware=hardware_info(dev), checks=[])
fails = []
for amp in ((True,) if preset_d.get("triton_mlp") else (False, True)):
    for n in steps:
        tr_v = {**tr, "amp": amp, "rollout": n, "chunk": n}
        seeds, c0 = iteration_seeds(tr_v, args.seed)
        w = loss_weights(tr["loss_window"], n, dev)
        r = compare_backends(model, EngineConfig(backend="reference", checkpoint_segment=None),
                             EngineConfig(**preset_d), tr_v, batch, seeds, c0, w)
        r.pop("grad_per_param")
        prec = "bf16" if amp else "fp32"
        ok, why = within_tolerance(r, DEFAULT_TOL[prec])
        if not ok:
            fails.append(f"preset:{prec}:{n}: {why}")
        report["checks"].append(dict(variant="preset", amp=amp, steps=n, ok=ok, why=why, **r))
        print(f"preset       {prec} {n:>4} steps: state {r['state_max_abs']:.2e} "
              f"(max {r['state_max_abs_ref']:.2f}), loss {r['loss_abs']:.2e}, "
              f"grad {r['grad_max_abs']:.2e} (max {r['grad_max_abs_ref']:.2e})  "
              + ("PASS" if ok else "FAIL " + why), flush=True)
if args.out:
    save_json(report, args.out)
print(("FAIL:\n  " + "\n  ".join(fails)) if fails else f"PASS: {len(report['checks'])} checks")
sys.exit(1 if fails else 0)
