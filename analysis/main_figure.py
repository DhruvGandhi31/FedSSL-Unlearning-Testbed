"""Main figure: forget-class accuracy gap vs retrain per unlearning method, SSL vs supervised,
mean ± std over seeds (gaps paired within a seed). Also writes the table as markdown and CSV.
python -m analysis.main_figure [--out results/figures]"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = {"ssl": "excl_class90_ssl", "sup": "excl_class90_sup"}
# (unlearn dir prefix, label). The supervised arm has no "original" labeler variants (lambda_u = 0).
METHODS = [("original", "original (no unlearning)"), ("federaser-current", "FedEraser\nlabeler: current"),
           ("federaser-original", "FedEraser\nlabeler: original"), ("pga-current", "PGA\nlabeler: current"),
           ("pga-original", "PGA\nlabeler: original")]
COLORS = {"ssl": "#2a78d6", "sup": "#eb6834"}  # categorical slots 1-2, validated on the light surface
NAMES = {"ssl": "semi-supervised (FixMatch)", "sup": "supervised (λ_u = 0)"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def final_eval(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return [e for e in (json.loads(l).get("eval") for l in f) if e][-1]


def class0_pls(ev: dict, k: int = 0, forget: int = 0) -> int:
    return sum(s["pl_count"][k] for s in ev.get("pl_scan", []) if s["cid"] != forget)


def collect() -> dict:
    """{arm: {seed: {key: (class-k acc, class-k PLs on other clients, test acc)}}} for finished runs."""
    out: dict = {}
    for arm, name in ARMS.items():
        for run in sorted(Path("results", name).glob("s*-*")):
            if not (run / "final.pt").exists():
                continue
            seed = int(run.name[1:].split("-")[0])
            ev = final_eval(run / "metrics.jsonl")
            models = {"original": ev}
            for d in sorted((run / "unlearn").glob("*")):  # sorted: the latest finished run of a method wins
                if (d / "final.pt").exists():
                    models["retrain" if d.name.startswith("retrain-") else d.name.rsplit("-", 2)[0]] = \
                        final_eval(d / "metrics.jsonl")
            out.setdefault(arm, {})[seed] = {m: (e["class_acc"][0], class0_pls(e), e["test_acc"])
                                             for m, e in models.items()}
    return out


def summarize(data: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        for key, label in METHODS:
            seeds = [s for s, m in data.get(arm, {}).items() if key in m and "retrain" in m]
            if not seeds:
                continue
            gap = [data[arm][s][key][0] - data[arm][s]["retrain"][0] for s in seeds]
            pls = [data[arm][s][key][1] for s in seeds]
            test = [data[arm][s][key][2] for s in seeds]
            sd = lambda xs: st.stdev(xs) if len(xs) > 1 else 0.0
            rows.append({"arm": arm, "method": key, "label": label.replace("\n", ", "), "n_seeds": len(seeds),
                         "gap_mean": st.mean(gap), "gap_std": sd(gap), "pls_mean": st.mean(pls), "pls_std": sd(pls),
                         "test_mean": st.mean(test), "test_std": sd(test),
                         "retrain_cls0_mean": st.mean(data[arm][s]["retrain"][0] for s in seeds),
                         "retrain_pls_mean": st.mean(data[arm][s]["retrain"][1] for s in seeds)})
    return rows


def plot(rows: list[dict], out: Path) -> None:
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)
    width = 0.36
    for ax, (field, title, ylabel) in zip(axes, [
        ("gap", "Forget-class accuracy gap to retrain (0 = perfect unlearning)", "class-0 test acc − retrain's"),
        ("pls", "Propagation: confident class-0 pseudo-labels on clients 1–9", "count (final model)")]):
        ax.set_facecolor(SURFACE)
        for i, arm in enumerate(ARMS):
            for j, (key, _) in enumerate(METHODS):
                r = next((r for r in rows if r["arm"] == arm and r["method"] == key), None)
                if r is None:
                    continue
                x = j + (i - 0.5) * width
                ax.bar(x, r[f"{field}_mean"], width * 0.92, color=COLORS[arm], zorder=2,
                       label=f"{NAMES[arm]}" if j == 0 or (key == "federaser-current" and arm == "sup") else None)
                if r["n_seeds"] > 1:
                    ax.errorbar(x, r[f"{field}_mean"], yerr=r[f"{field}_std"], color=INK2, lw=1, capsize=3, zorder=3)
        if field == "pls":  # retrain's level as a reference line, per arm
            for arm in ARMS:
                rr = [r for r in rows if r["arm"] == arm]
                if rr:
                    ax.axhline(rr[0]["retrain_pls_mean"], color=COLORS[arm], lw=1, ls="--", zorder=1)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        ax.set_xticks(range(len(METHODS)), [m[1] for m in METHODS], fontsize=9)
        ax.set_title(title, fontsize=11, loc="left", color=INK)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    n = max((r["n_seeds"] for r in rows), default=0)
    fig.legend(seen.values(), seen.keys(), loc="upper right", frameon=False, ncol=2)
    fig.suptitle(f"Unlearning client 0 (90% of class-0 labels) — mean ± std over {n} seed(s); "
                 "dashed = retrain's pseudo-label count", x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"main_figure.{ext}", dpi=200, facecolor=SURFACE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect()
    rows = summarize(data)
    with open(out / "main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| arm | model | seeds | class-0 gap vs retrain | class-0 PLs on clients 1–9 | test acc |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['label']} | {r['n_seeds']} | {r['gap_mean']:+.3f} ± {r['gap_std']:.3f} | "
                     f"{r['pls_mean']:.0f} ± {r['pls_std']:.0f} (retrain {r['retrain_pls_mean']:.0f}) | "
                     f"{r['test_mean']:.3f} ± {r['test_std']:.3f} |")
    (out / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    plot(rows, out)
    print(f"wrote {out / 'main_figure.png'}, .pdf, main_table.md, main_table.csv")


if __name__ == "__main__":
    main()
