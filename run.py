"""Train one federated run: python run.py --config configs/xxx.yaml [--seed N]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fedssl.config import load_config, save_config
from fedssl.data import class_counts, load_data
from fedssl.metrics import MetricsWriter, git_commit
from fedssl.models import WideResNet
from fedssl.server import fedavg, make_schedule, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="overrides the config's training seed")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_everything(cfg.seed)

    run_dir = Path(cfg.out_dir) / cfg.name / f"s{cfg.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")

    data, clients = load_data(cfg.data, device)
    y_train = data.y_train.cpu().numpy()
    partition = [{"cid": c.cid, "labeled": class_counts(c.labeled, y_train).tolist(),
                  "unlabeled": class_counts(c.unlabeled, y_train).tolist()} for c in clients]
    (run_dir / "partition.json").write_text(json.dumps(partition))

    schedule = make_schedule(cfg.data.num_clients, cfg.fed.rounds, cfg.fed.participation, cfg.seed)
    (run_dir / "schedule.json").write_text(json.dumps(schedule))

    model = WideResNet(bn_momentum=cfg.train.bn_momentum).to(device)
    torch.save(model.state_dict(), run_dir / "init.pt")

    log = MetricsWriter(run_dir / "metrics.jsonl")
    log.write({"header": True, "git_commit": git_commit(), "device": str(device),
               "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(f"run dir: {run_dir}", flush=True)
    t0 = time.time()
    fedavg(cfg, data, clients, schedule, model, run_dir, log)
    log.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
