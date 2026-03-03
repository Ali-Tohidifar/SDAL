import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class WorkerResult:
    worker_id: int
    cmd: List[str]
    cwd: str
    dataset_dir: Path
    logs_dir: Path
    stdout_path: Path
    stderr_path: Path
    start_time: str
    end_time: Optional[str]
    duration_s: Optional[float]
    exit_code: Optional[int]
    num_sequences: Optional[int]
    num_jpg: Optional[int]


def _count_outputs(dataset_dir: Path) -> Tuple[int, int]:
    if not dataset_dir.exists():
        return 0, 0
    seqs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    jpg = list(dataset_dir.rglob("*.jpg"))
    return len(seqs), len(jpg)


def run_parallel_generators(
    *,
    blender_bin: str,
    empty_blend: Path,
    generator_py: Path,
    config_json: Path,
    run_root: Path,
    data_gen_env_dir: Path,
    num_workers: int,
    logger=None,
    poll_s: float = 5.0,
) -> Dict:
    """
    Launch multiple Blender data generation workers in parallel.

    Each worker writes to:
      - <run_root>/worker_XX/Dataset/...
      - <run_root>/worker_XX/logs/...
      - stdout/stderr captured into that logs folder

    Returns a manifest dict and also writes it to <run_root>/manifest.json.
    """
    # Make run_root absolute so dataset/log paths passed to Blender are absolute too.
    # This avoids accidental outputs under data_gen_env_dir when we run Blender with cwd=data_gen_env_dir.
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    empty_blend = Path(empty_blend).resolve()
    generator_py = Path(generator_py).resolve()
    config_json = Path(config_json).resolve()
    data_gen_env_dir = Path(data_gen_env_dir).resolve()

    started_at = datetime.now().isoformat()
    workers: List[WorkerResult] = []
    procs: List[Dict] = []

    for wid in range(1, num_workers + 1):
        wdir = run_root / f"worker_{wid:02d}"
        dataset_dir = (wdir / "Dataset").resolve()
        logs_dir = (wdir / "logs").resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = logs_dir / "stdout.txt"
        stderr_path = logs_dir / "stderr.txt"

        cmd = [
            str(blender_bin),
            str(empty_blend),
            "--background",
            "--python",
            str(generator_py),
            "--",
            "--config-json",
            str(config_json),
            "--dataset-dir",
            str(dataset_dir),
            "--logs-dir",
            str(logs_dir),
            "--worker-id",
            str(wid),
        ]

        start_time = datetime.now().isoformat()
        wr = WorkerResult(
            worker_id=wid,
            cmd=cmd,
            cwd=str(data_gen_env_dir),
            dataset_dir=dataset_dir,
            logs_dir=logs_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            start_time=start_time,
            end_time=None,
            duration_s=None,
            exit_code=None,
            num_sequences=None,
            num_jpg=None,
        )
        workers.append(wr)

        # Ensure child processes find Avatars/Scenes/Horizon/Empty via CWD
        # and avoid inheriting a weird python env var.
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")

        if logger:
            logger.info(f"[synthgen] starting worker {wid:02d}: {' '.join(cmd)} (cwd={data_gen_env_dir})")

        stdout_f = open(stdout_path, "w", encoding="utf-8")
        stderr_f = open(stderr_path, "w", encoding="utf-8")
        p = subprocess.Popen(
            cmd,
            cwd=str(data_gen_env_dir),
            stdout=stdout_f,
            stderr=stderr_f,
            env=env,
            text=True,
        )
        procs.append({"proc": p, "wr": wr, "stdout_f": stdout_f, "stderr_f": stderr_f})

    # Wait for all workers
    while True:
        alive = 0
        for entry in procs:
            p: subprocess.Popen = entry["proc"]
            wr: WorkerResult = entry["wr"]
            rc = p.poll()
            if rc is None:
                alive += 1
            else:
                if wr.exit_code is None:
                    wr.exit_code = int(rc)
                    wr.end_time = datetime.now().isoformat()
                    wr.duration_s = time.time() - datetime.fromisoformat(wr.start_time).timestamp()
                    wr.num_sequences, wr.num_jpg = _count_outputs(wr.dataset_dir)
                    # Close file handles
                    try:
                        entry["stdout_f"].close()
                        entry["stderr_f"].close()
                    except (OSError, ValueError):
                        pass
                    if logger:
                        logger.info(
                            f"[synthgen] worker {wr.worker_id:02d} finished rc={wr.exit_code} "
                            f"seqs={wr.num_sequences} jpg={wr.num_jpg} logs={wr.logs_dir}"
                        )
        if alive == 0:
            break
        time.sleep(poll_s)

    ended_at = datetime.now().isoformat()

    manifest = {
        "started_at": started_at,
        "ended_at": ended_at,
        "num_workers": num_workers,
        "run_root": str(run_root),
        "config_json": str(config_json),
        "empty_blend": str(empty_blend),
        "generator_py": str(generator_py),
        "data_gen_env_dir": str(data_gen_env_dir),
        "workers": [
            {
                "worker_id": w.worker_id,
                "cmd": w.cmd,
                "cwd": w.cwd,
                "dataset_dir": str(w.dataset_dir),
                "logs_dir": str(w.logs_dir),
                "stdout": str(w.stdout_path),
                "stderr": str(w.stderr_path),
                "start_time": w.start_time,
                "end_time": w.end_time,
                "duration_s": w.duration_s,
                "exit_code": w.exit_code,
                "num_sequences": w.num_sequences,
                "num_jpg": w.num_jpg,
            }
            for w in workers
        ],
        "any_failed": any((w.exit_code or 0) != 0 for w in workers),
        "total_sequences": sum((w.num_sequences or 0) for w in workers),
        "total_jpg": sum((w.num_jpg or 0) for w in workers),
    }

    manifest_path = run_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if logger:
        logger.info(f"[synthgen] manifest written: {manifest_path}")

    return manifest

