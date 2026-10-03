"""Final experiment: original run, retrain and unlearning for the SSL and supervised arms, over seeds.
python run_final.py [--seeds 0 1 2] [--workers 3]
Resumable: a step whose final.pt already exists is skipped, so rerunning picks up where it stopped."""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from fedssl.config import load_config

ARMS = {"ssl": "configs/excl_class_ssl.yaml", "sup": "configs/excl_class_sup.yaml"}
# (method, labeler). With lambda_u = 0 the labeler only affects logging, so the supervised arm runs one variant.
METHODS = {
    "ssl": [("retrain", "current"), ("federaser", "current"), ("federaser", "original"),
            ("pga", "current"), ("pga", "original")],
    "sup": [("retrain", "current"), ("federaser", "current"), ("pga", "current")],
}
LOG_DIR = Path("results/final_logs")
_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(LOG_DIR / "progress.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def finished_run(name: str, seed: int) -> Path | None:
    done = [r for r in sorted(Path("results", name).glob(f"s{seed}-*")) if (r / "final.pt").exists()]
    return done[-1] if done else None


def unlearn_done(run: Path, method: str, labeler: str) -> bool:
    prefix = "retrain-" if method == "retrain" else f"{method}-{labeler}-"
    return any((d / "final.pt").exists() for d in (run / "unlearn").glob(prefix + "*"))


def step(cmd: list[str], log: Path) -> bool:
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run([sys.executable, "-u", *cmd], stdout=f, stderr=subprocess.STDOUT).returncode == 0


def chain(arm: str, seed: int) -> None:
    """Original run, then every unlearning method on it."""
    cfg = ARMS[arm]
    name = load_config(cfg).name
    tag = f"{arm} s{seed}"
    run = finished_run(name, seed)
    if run is None:
        say(f"{tag}: original start")
        if not step(["run.py", "--config", cfg, "--seed", str(seed)], LOG_DIR / f"{arm}_s{seed}_original.log"):
            say(f"{tag}: original FAILED, skipping its unlearning")
            return
        run = finished_run(name, seed)
        say(f"{tag}: original done -> {run}")
    else:
        say(f"{tag}: original already done ({run})")
    for method, labeler in METHODS[arm]:
        what = f"{method}/{labeler}"
        if unlearn_done(run, method, labeler):
            say(f"{tag}: {what} already done")
            continue
        say(f"{tag}: {what} start")
        cmd = ["unlearn.py", "--run", str(run), "--method", method]
        if method != "retrain":
            cmd += ["--labeler", labeler]
        ok = step(cmd, LOG_DIR / f"{arm}_s{seed}_{method}_{labeler}.log")
        say(f"{tag}: {what} {'done' if ok else 'FAILED'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=3, help="chains run in parallel on the one GPU")
    parser.add_argument("--smoke", action="store_true", help="use configs/smoke.yaml for both arms")
    args = parser.parse_args()
    if args.smoke:
        ARMS.update(ssl="configs/smoke.yaml", sup="configs/smoke.yaml")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":  # keep the machine awake until this process exits (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    # The code the runs used: HEAD is in each metrics.jsonl header; record any uncommitted changes too.
    diff = subprocess.run(["git", "diff"], capture_output=True, text=True, encoding="utf-8").stdout
    (LOG_DIR / f"launch-{time.strftime('%Y%m%d-%H%M%S')}.diff").write_text(diff, encoding="utf-8")

    # Seed-major order: both arms of a seed finish before the next seed's chains start.
    jobs: queue.Queue = queue.Queue()
    for seed in args.seeds:
        for arm in ("ssl", "sup"):
            jobs.put((arm, seed))
    say(f"start: seeds {args.seeds}, {args.workers} workers, uncommitted diff {len(diff)} chars")

    def worker() -> None:
        while True:
            try:
                arm, seed = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                chain(arm, seed)
            except Exception as e:  # keep the other chains going
                say(f"{arm} s{seed}: CRASHED {e!r}")

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("all done")


if __name__ == "__main__":
    main()
