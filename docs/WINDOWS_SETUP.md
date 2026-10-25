# Windows setup, step by step

This guide takes a Windows 10 or 11 PC with an NVIDIA GPU (8 GB of VRAM is what the configs are
sized for) from nothing to a running phase 1. Every command is for PowerShell, run from the
repository root unless said otherwise.

## 1. Install Python 3.11 or newer

Download the installer from https://www.python.org/downloads/windows/ (3.11 or 3.12), run it,
tick **"Add python.exe to PATH"** on the first screen, then **Install Now**. Open a new
PowerShell window and check:

```powershell
py -3 --version
```

If `py` is not found, use `python --version`; if that is not found either, the PATH box was not
ticked: rerun the installer and choose "Modify".

## 2. Install the NVIDIA driver

A recent driver is all that is needed; the CUDA runtime comes with the PyTorch wheel. Update from
GeForce Experience or https://www.nvidia.com/drivers, reboot, then:

```powershell
nvidia-smi
```

should print your GPU, the driver version and the CUDA version it supports (12.4 or newer is
needed for the `cu124` wheel below; an older driver works with the `cu121` wheel).

## 3. Get the repository and create a virtual environment

```powershell
git clone <repository url> ncas
cd ncas
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is refused ("running scripts is disabled on this system"), run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

and activate again. The prompt now starts with `(.venv)`. Do this activation in every new
PowerShell window before running anything.

## 4. Install PyTorch with CUDA, then the package

```powershell
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

The first line downloads about 2.5 GB. Check that the GPU is visible:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

It must print `True` and your GPU name. If it prints `False`:

* `pip show torch` says a version without `+cu124`: you got the CPU wheel. Run
  `pip uninstall torch` then the `pip install torch --index-url ...` line again.
* The driver is too old for CUDA 12.4: install the `cu121` wheel instead
  (`--index-url https://download.pytorch.org/whl/cu121`) or update the driver.
* A laptop with two GPUs: make sure the NVIDIA one is selected for python.exe in the NVIDIA
  Control Panel ("Manage 3D settings" > "Program settings").

## 5. Run the tests and the smoke run

```powershell
pytest -q
python scripts\train.py --config configs\smoke_maze.yaml --device cpu
```

The tests take a few minutes on CPU and need no network. The smoke run takes about a minute and
writes `runs\smoke_maze\seed0\` (see [TUTORIAL.md](TUTORIAL.md), level 0). Now the same on the
GPU:

```powershell
python scripts\train.py --config configs\smoke_maze.yaml
```

(the default `--device auto` picks CUDA when available). Both forward and backward slashes work
in these paths; the phase scripts use forward slashes internally.

## 6. Run the phase scripts

The `.ps1` scripts are the Windows twins of the `.sh` ones, with the same arguments:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 maze       # Maze-OOD only, about 6.5 h (measured, RTX 4060 Laptop)
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1            # all of phase 1, about 2 days
powershell -ExecutionPolicy Bypass -File scripts\run_phase1.ps1 sudoku     # Sudoku-OOD only, about 45 h
powershell -ExecutionPolicy Bypass -File scripts\run_phase2.ps1            # about 25 h  (or: maze_hard | sudoku_extreme)
powershell -ExecutionPolicy Bypass -File scripts\run_phase3.ps1            # about 6 h
```

Run them from the repository root with the virtual environment activated (the scripts `cd` to the
repository root themselves, but `python` must be the one of the venv). Each script stops at the
first failure (`$ErrorActionPreference = "Stop"`), prints every command it runs, and skips the
steps that are already done: a finished training is a no-op and an evaluation whose JSON exists
is skipped (`skip (exists): results\...`).

Keep the PowerShell window open while a phase runs. To keep working on the PC meanwhile, open a
second window; the GPU is shared, so avoid other GPU work. Watch the GPU with
`nvidia-smi -l 5` in another window (VRAM use, utilisation, temperature).

## 7. Keep the PC awake

A phase is interrupted when Windows sleeps (hibernation also stops it; a locked screen does not).
Either:

* Settings > System > Power & battery (Power & sleep on Windows 10) > Screen and sleep: set
  "When plugged in, put my device to sleep after" to **Never**. The screen can still turn off.
* Or from an administrator PowerShell, for the current power plan:

  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  ```

  (`powercfg /change standby-timeout-ac 30` restores a 30-minute timeout afterwards).
* On a laptop, also set "When I close the lid" to "Do nothing" (Control Panel > Power Options >
  Choose what closing the lid does) and keep it plugged in.
* Windows Update can reboot the machine at night: pause updates for a week (Settings > Windows
  Update > Pause updates) or set active hours to cover the run.

If the machine does sleep or reboots anyway, nothing is lost beyond the last checkpoint: rerun
the same phase command and it resumes (next section).

## 8. Where things land

| folder | content |
|---|---|
| `data\` | datasets, downloaded or generated on first use (see [../data/README.md](../data/README.md)); gitignored |
| `runs\<config>\seed<k>\` | one folder per training: `run.json`, `log.jsonl`, `checkpoint.pt`, `final.pt`, `summary.json`; ablations in `runs\ablations_<task>\<variant>\seed<k>\` |
| `results\<experiment>\<name>_seed<k>.json` | one JSON per evaluation and seed; `results\summary.json` from `aggregate.py` |
| `figures\F*.png`, `figures\F*.pdf` | from `make_figures.py`; `figures\anim\*.gif` and `*_strip.png` from `animate.py` |

Disk use is modest: the official Maze-OOD test archives are the largest downloads (a few hundred
MB for all sizes); `checkpoint.pt` of Sudoku-Extreme is a few tens of MB. Windows Defender may
scan every file written to `data\` and `runs\`; if the data generation seems slow, add the
repository folder to the exclusions (Windows Security > Virus & threat protection > Exclusions).

## 9. How to resume

* **A training was interrupted** (Ctrl+C, sleep, crash): run the same `train.py` command or the
  same phase script again. It prints `resumed from step N` and continues from `checkpoint.pt`
  (saved every 500 steps for the maze and ARC configs, every 1,000 for Sudoku). The checkpoint
  holds the model, the EMA, the optimizer, the replay pool and the random states.
* **An evaluation was interrupted**: evaluations are not checkpointed; rerun the phase script and
  it redoes only the evaluations whose JSON is missing. To redo one that exists, delete its JSON
  first.
* **Start over**: delete `runs\<config>\seed<k>\` (or use `--no-resume` on `train.py`, which
  overwrites in place; `--out` picks another folder).
* **Change the budget of a run**: `python scripts\train.py --config configs\sudoku_ood.yaml --seed 0 --set train.steps=20000`
  trains 20,000 steps instead of 50,000 (and `--set` can be repeated). Record such changes in
  your own copy of DEVIATIONS.md.
* **CUDA out of memory**: see "Memory tips for 8 GB" in [TUTORIAL.md](TUTORIAL.md)
  (`--set train.chunk=...`, `--set train.amp=true`, `--max-cells` on the eval scripts). Close
  other GPU programs (browsers with hardware acceleration can hold a few hundred MB).
* **Force the CPU**: add `--device cpu` to any script (only sensible for the smoke config and
  for debugging).

## 10. Useful one-liners

```powershell
# last training line and last evaluation of a run
Get-Content runs\maze_ood\seed0\log.jsonl | Select-String '"loss"' | Select-Object -Last 1
Get-Content runs\maze_ood\seed0\log.jsonl | Select-String '"eval"' | Select-Object -Last 1

# the headline table so far
python scripts\aggregate.py

# redraw the figures that have data
python scripts\make_figures.py

# help of any script
python scripts\eval_tts.py --help
```
