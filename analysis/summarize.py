"""Markdown summary of a run and its unlearning outputs.
python analysis/summarize.py results/<name>/<run> [--diagnose]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def final_eval(path: Path) -> dict | None:
    evals = [l["eval"] for l in read_jsonl(path) if "eval" in l]
    return evals[-1] if evals else None


def propagation(ev: dict, k: int, forget: int) -> str:
    """Confident class-k pseudo-labels on clients other than the forget client (and how many are correct)."""
    if "pl_scan" not in ev:
        return "n/a"
    scan = [s for s in ev["pl_scan"] if s["cid"] != forget]
    return f"{sum(s['pl_count'][k] for s in scan)} ({sum(s['pl_correct'][k] for s in scan)} correct)"


def row(name: str, ev: dict, k: int, forget: int, ref_k: float | None) -> str:
    gap = "" if ref_k is None else f"{ev['class_acc'][k] - ref_k:+.4f}"
    return f"| {name} | {ev['test_acc']:.4f} | {ev['class_acc'][k]:.4f} | {gap} | {propagation(ev, k, forget)} |"


def diagnose(run: Path, k: int, forget: int) -> list[str]:
    """Where class-k test images go under the final model, and whether one round of the forget
    client's own local training (from the final model, last-round lr) learns class k."""
    import copy

    from fedssl.client import local_train
    from fedssl.config import load_config
    from fedssl.data import load_data
    from fedssl.metrics import evaluate
    from fedssl.models import WideResNet
    from fedssl.server import round_lr, seed_everything

    cfg = load_config(run / "config.yaml")
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, clients = load_data(cfg.data, device)
    model = WideResNet(bn_momentum=cfg.train.bn_momentum).to(device)
    final = torch.load(run / "final.pt", map_location=device)
    model.load_state_dict(final)
    with torch.no_grad():
        model.eval()
        xk = data.x_test[data.y_test == k]
        pred = torch.cat([model(xk[i:i + 1000].float() / 255).argmax(1) for i in range(0, len(xk), 1000)])
    out = [f"- Final model, class-{k} test images predicted as: {torch.bincount(pred, minlength=10).tolist()}"]

    labeler = copy.deepcopy(model)
    local_train(final, clients[forget], cfg.train, labeler, model=model, data=data,
                lr=round_lr(cfg, cfg.fed.rounds - 1))
    ev = evaluate(model, data)
    out.append(f"- One round of client {forget}'s local training from the final model: class-{k} acc "
               f"{ev['class_acc'][k]:.4f}, test acc {ev['test_acc']:.4f}, "
               f"per-class {[round(a, 2) for a in ev['class_acc']]}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)

    import yaml
    with open(run / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    k, forget = cfg["data"]["forget_class"], cfg.get("unlearn", {}).get("forget_client", 0)

    unlearned = {}
    for d in sorted((run / "unlearn").glob("*")) if (run / "unlearn").exists() else []:
        ev = final_eval(d / "metrics.jsonl") if (d / "metrics.jsonl").exists() else None
        if ev is not None and (d / "final.pt").exists():  # finished runs only
            unlearned[d.name] = ev
    retrain = [ev for name, ev in unlearned.items() if name.startswith("retrain")]
    ref_k = retrain[-1]["class_acc"][k] if retrain else None

    print(f"| model | test acc | class-{k} acc | class-{k} gap vs retrain | "
          f"confident class-{k} PLs on clients != {forget} |")
    print("|---|---|---|---|---|")
    print(row("original", final_eval(run / "metrics.jsonl"), k, forget, ref_k))
    for name, ev in unlearned.items():
        print(row(name, ev, k, forget, ref_k))

    print(f"\nclass-{k} accuracy over training (eval rounds):")
    print(", ".join(f"r{l['round']}: {l['eval']['class_acc'][k]:.3f}"
                    for l in read_jsonl(run / "metrics.jsonl") if "eval" in l and (l["round"] + 1) % 25 == 0))
    if args.diagnose:
        print("\n" + "\n".join(diagnose(run, k, forget)))


if __name__ == "__main__":
    main()
