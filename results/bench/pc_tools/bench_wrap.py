"""scripts/bench_engine.py unchanged, with TF32 off, an optional `preset` variant (the engine:
section of --preset-config, without its checkpoint_segment) and the driver version in the JSON.

    python bench_wrap.py [--preset-config configs/l4_round3/sudoku_ood.yaml] <bench_engine args>
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import bench_engine  # noqa: E402
from nca.engine.engine import EngineConfig  # noqa: E402
from nca.utils import load_config  # noqa: E402

argv = sys.argv[1:]
preset_path = None
if "--preset-config" in argv:
    i = argv.index("--preset-config")
    preset_path = argv[i + 1]
    del argv[i : i + 2]
    preset_sets = []
    while "--preset-set" in argv:
        j = argv.index("--preset-set")
        preset_sets.append(argv[j + 1])
        del argv[j : j + 2]
    d = dataclasses.asdict(EngineConfig.from_dict(load_config(preset_path, preset_sets)["engine"]))
    d.pop("checkpoint_segment", None)
    bench_engine.VARIANTS["preset"] = d
    print("preset engine:", json.dumps(d, default=str), flush=True)
sys.argv = ["bench_engine.py", *argv]
bench_engine.main()

out = argv[argv.index("--out") + 1] if "--out" in argv else None
if out:
    p = Path(out)
    rep = json.loads(p.read_text())
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        drv = f"unknown ({e})"
    rep["hardware"]["driver"] = drv
    rep["tf32"] = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
    rep["preset_config"] = preset_path
    rep["preset_overrides"] = preset_sets if preset_path else None
    rep["wrapper"] = "bench_wrap.py: bench_engine.main() with TF32 off and a 'preset' variant"
    p.write_text(json.dumps(rep, indent=2, default=str))
