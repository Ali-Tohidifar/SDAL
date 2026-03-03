# SDAL Mode-B — Docker-free Synthetic Generation (5× parallel Blender)

This document describes the refactored **Docker-free** synthetic data generation pipeline used by:

- `SDAL/SDAL_modeB.py` (Mode-B day-by-day)
- `SDAL/SDAL.py` (cycle-based SDAL)

It replaces the old Docker-based `run_blendcon.sh` path with **direct Blender subprocess** execution and a **run-scoped output layout**.

## What gets generated (two layers)

### 1) Final YOLO training data (used by YOLO training)

This is what the SDAL trainers consume.

- **Mode-B**: `SDAL/experiments/modeB_runs/<run_id>/dayXX/synth_train/`
  - `images/` (JPEGs copied from raw sequences)
  - `labels/` (YOLO `.txt` files generated from `Joint_Tracker.pickle`)

- **SDAL.py**: the `train:` path in `paths['synth_data_yaml']` (commonly `datasets/synth/`)
  - `datasets/synth/images/`
  - `datasets/synth/labels/`

### 2) Raw generator artifacts + logs (for debugging)

These are run-scoped per failure case and per worker.

- **Mode-B**: `SDAL/experiments/modeB_runs/<run_id>/dayXX/oracle_runs/<failure_case_id>/`
  - `config.json` (normalized config consumed by Blender)
  - `manifest.json` (exit codes + counts per worker)
  - `worker_01..worker_05/`
    - `Dataset/` (raw sequence folders from Blender; each should contain `*.jpg` and `Joint_Tracker.pickle`)
    - `logs/`
      - `stdout.txt`, `stderr.txt` (captured subprocess streams)
      - `data_generation_*.log` (Blender-script internal log)

## Hyperparameters / config knobs

In Mode-B, see `SDAL/cfg/modeB_hyp.yaml`:

- `num_containers`: number of parallel Blender workers (we keep the name for backward compatibility). Set to `5`.
- `blender_bin`: path to Blender executable (default: `/snap/bin/blender`).
- `keep_oracle_artifacts`:
  - `false` (default): after YOLO conversion, delete each `worker_XX/Dataset/` but keep logs + manifest.
  - `true`: keep the full raw datasets for inspection.

For `SDAL.py`, add the same keys to your chosen hyp yaml (e.g. `cfg/hyp_small_test.yaml`).

## How the pipeline runs (high-level)

1. **Decomposer / retrieval** (Python): finds similar scene + avatars and writes `config.json` into the run folder.
2. **Normalization** (Python calls Blender): updates normalized positions inside the same `config.json`.
3. **Parallel generation** (5× Blender processes): each worker renders sequences into its own `worker_XX/Dataset/`.
4. **YOLO conversion**:
   - `pickle2yolo(...)` reads `Joint_Tracker.pickle` inside each sequence folder.
   - Images and `.txt` labels are copied into the YOLO dataset destination.

Key code paths:

- Orchestrator: `SDAL/synthetic_oracle.py` (`oracle(...)`)
- Parallel runner: `SDAL/sdal_utils/blender_parallel_runner.py`
- Blender generator: `SDAL/sdal_utils/Data_Generator/231109_Adaptive_Data_Generator.py`
- Normalization: `SDAL/blender_depended_codes/normalize_avatar_location.py`

## Logging (where to look when something fails)

For a single failure case:

- Start here: `<run_root>/manifest.json`
  - `exit_code` per worker
  - `stdout`/`stderr` paths per worker
  - counts (`num_sequences`, `num_jpg`)

Then inspect per worker:

- `<run_root>/worker_XX/logs/stderr.txt`
- `<run_root>/worker_XX/logs/stdout.txt`
- `<run_root>/worker_XX/logs/data_generation_*.log`

If you see **images but no `Joint_Tracker.pickle`**, YOLO conversion will fail and the pipeline will skip copying those sequences.

## Important gotcha: absolute paths vs relative paths

Blender workers are launched with `cwd=SDAL/sdal_utils/Data_Generator` so they can find `Avatars/`, `Scenes/`, `Empty.blend`, etc.

Because of that, **all output directories passed to Blender must be absolute**; otherwise Blender will write outputs under `sdal_utils/Data_Generator/...` accidentally.

If you already have a stray folder like:

- `SDAL/sdal_utils/Data_Generator/experiments/...`

it was created by an older run that used relative output paths. It can be deleted safely (it is only raw artifacts/logs).

## Starting a fresh experiment (cleanup)

### Mode-B (recommended)

To fully reset one run:

- Delete the run folder:
  - `SDAL/experiments/modeB_runs/run_<run_id>/`

Optional additional cleanup:

- If you kept raw artifacts and want to reclaim space:
  - delete `.../dayXX/oracle_runs/`

### SDAL.py

To reset synthetic dataset outputs:

- Delete:
  - `datasets/synth/images/`
  - `datasets/synth/labels/`

Also remove any YOLO cache files if you see stale entries:

- `find datasets -name \"*.cache\" -delete`

