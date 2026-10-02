"""Unlearn the forget client from a finished run:
python unlearn.py --run results/<name>/<run> --method retrain|federaser|pga [--labeler current|original|none|PATH]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from fedssl.config import load_config, save_config
from fedssl.data import load_data
from fedssl.metrics import MetricsWriter, git_commit
from fedssl.models import WideResNet
from fedssl.server import GLOBAL, seed_everything
from fedssl.unlearn.federaser import federaser
from fedssl.unlearn.pga import pga
from fedssl.unlearn.retrain import retrain


def resolve_labeler(arg: str, run_dir: Path, bn_momentum: float, device: torch.device) -> nn.Module | None | str:
    """current: the model being unlearned/rebuilt; original: the run's (contaminated) final model;
    none: no pseudo-labels; anything else: path to a state_dict, e.g. a retrain's final.pt."""
    if arg == "current":
        return GLOBAL
    if arg == "none":
        return None
    path = run_dir / "final.pt" if arg == "original" else Path(arg)
    model = WideResNet(bn_momentum=bn_momentum).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run directory written by run.py")
    parser.add_argument("--method", required=True, choices=["retrain", "federaser", "pga"])
    parser.add_argument("--labeler", default="current",
                        help="pseudo-labeler for calibration/recovery: current | original | none | PATH")
    args = parser.parse_args()
    if args.method == "retrain" and args.labeler != "current":
        parser.error("retrain always pseudo-labels with its own global model")

    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_everything(cfg.seed)

    tag = args.method if args.method == "retrain" else f"{args.method}-{Path(args.labeler).stem}"
    out_dir = run_dir / "unlearn" / f"{tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True)
    save_config(cfg, out_dir / "config.yaml")
    (out_dir / "args.json").write_text(json.dumps(vars(args)))

    data, clients = load_data(cfg.data, device)
    schedule = json.loads((run_dir / "schedule.json").read_text())
    model = WideResNet(bn_momentum=cfg.train.bn_momentum).to(device)
    load = lambda name: torch.load(run_dir / name, map_location=device)
    labeler = resolve_labeler(args.labeler, run_dir, cfg.train.bn_momentum, device)

    log = MetricsWriter(out_dir / "metrics.jsonl")
    log.write({"header": True, "git_commit": git_commit(), "method": args.method, "labeler": args.labeler,
               "source_run": str(run_dir), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(f"out dir: {out_dir}", flush=True)
    t0 = time.time()
    if args.method == "retrain":
        weights = retrain(cfg, data, clients, schedule, load("init.pt"), model, out_dir, log)
    elif args.method == "federaser":
        weights = federaser(cfg, data, clients, schedule, load("init.pt"), run_dir / "history", model, log, labeler)
    else:
        weights = pga(cfg, data, clients, schedule, load("final.pt"), load("last_updates.pt"),
                      model, out_dir, log, labeler)
    torch.save(weights, out_dir / "final.pt")
    log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
