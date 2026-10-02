# FedSSL Unlearning Testbed

A small, single-machine research testbed for **federated semi-supervised learning (FedSSL)** and **federated unlearning (FU)**.

It asks one question:

> Does influence from a forgotten client survive standard federated unlearning **more** when training used unlabeled data, because that client's knowledge spread into other clients' pseudo-labels?

The target output is one figure: how far each unlearning method's forget metric is from retraining, **supervised vs. semi-supervised**.

## Status

| Phase | Contents | State |
|---|---|---|
| 1. Foundations | config, CIFAR-10 partitioning, WRN-28-2 | done |
| 2. Local learner | GPU augmentations, FixMatch loss, `local_train` | done |
| 3. Federation | FedAvg server, sampling schedule, update history, `run.py` | done |
| 4. Unlearning | retrain, FedEraser, projected gradient ascent | built; first full-run results pending |
| 5. Experiments | SSL vs. supervised configs, main figure | configs written |

## Setup

Requires Python 3.10+ and a CUDA GPU. Developed on an RTX 4060 Laptop GPU (8 GB) on Windows.

```
pip install -r requirements.txt
```

CIFAR-10 downloads to `data/` on first use (~170 MB).

## Usage

```
python run.py --config configs/smoke.yaml          # end-to-end check, < 1 min; run after any change to fedssl/
python run.py --config configs/X.yaml --seed 1     # --seed overrides the training seed
python -m fedssl.data --config X.yaml              # print the per-client class split
python unlearn.py --run results/<name>/<run> --method retrain|federaser|pga [--labeler L]
python -m pytest tests/ -q
```

Each run writes to `results/<name>/s<seed>-<timestamp>/`:

| File | Contents |
|---|---|
| `config.yaml` | resolved config |
| `metrics.jsonl` | header line (git commit, device, torch version), then one JSON object per round: participants, lr, per-client training stats; on eval rounds also test accuracy, per-class accuracy and a pseudo-label scan of every client's unlabeled pool |
| `schedule.json` | client set per round, drawn once from the training seed |
| `partition.json` | per-client class counts, labeled and unlabeled |
| `init.pt`, `final.pt` | global model state dicts |
| `history/rXXXX_cYY.pt` | fp16 client updates every `history_interval` rounds (for FedEraser) |
| `last_updates.pt` | each client's most recent update (for PGA's reference model) |

Unlearning results go to `<run>/unlearn/<method>-<labeler>-<timestamp>/` with their own `metrics.jsonl` and `final.pt`.
`--labeler` chooses who pseudo-labels during FedEraser calibration and PGA recovery rounds:
`current` (the model being unlearned), `original` (the run's contaminated final model), `none`, or a path to a state dict such as a retrain's `final.pt`.

## Layout

```
fedssl/
  config.py   # dataclass configs loaded from YAML; unknown keys and wrong types raise
  data.py     # CIFAR-10, Dirichlet partition, labeled/unlabeled split, forget scenarios
  models.py   # WRN-28-2 (~1.47M params), takes [0, 1] images and normalizes internally
  ssl.py      # batched on-GPU weak/strong augmentation, FixMatch loss
  client.py   # local_train(global_weights, client, cfg, pseudo_labeler, ...) -> update, stats
  server.py   # FedAvg loop, sampling schedule, update history
  metrics.py  # test / per-class accuracy, pseudo-label scan, metrics.jsonl writer
  unlearn/
    retrain.py    # same init, seed and schedule, forget client removed
    federaser.py  # rebuild from stored updates with calibration training
    pga.py        # projected gradient ascent + recovery rounds
configs/      # one YAML per experiment, plus smoke.yaml
run.py        # train a federated run
unlearn.py    # unlearn client 0 from a finished run
tests/
```

## Experimental setup

- **Data:** CIFAR-10 split over 10 clients with a Dirichlet(α=0.3) class split. Each client keeps 10% of its data labeled, stratified by class. The rest is unlabeled. True labels of unlabeled data are kept only for logging and never used in training. The partition depends only on `data_seed`, so the original and retrained runs see identical client data.
- **Local learner:** FixMatch with confidence threshold τ=0.95, μ=3, λ_u=1, and a fixed number of local steps per round.
- **Federation:** FedAvg with equal client weights, since every client runs the same number of local steps. 50% of clients take part each round. The learning rate follows FixMatch's cosine schedule over rounds.
- **Supervised control:** the same config with `lambda_u: 0`.
- **Forget scenario (exclusive class):** client 0 is the forgotten client. It gets 20% of class *k*'s images and holds *all* the labeled class-*k* examples (100 by default). The rest of class *k* goes to clients 1–9 as unlabeled data only. Any confident class-*k* pseudo-label on those clients is knowledge that came from client 0. The forget metric is test accuracy on class *k*, which should be about 0 after retraining without client 0.

## Unlearning methods

- **retrain:** FedAvg from the run's `init.pt` with the same seed, on the saved schedule with client 0 removed from every round. This is the reference the other methods are measured against.
- **FedEraser** (Liu et al. 2021): rebuilds the model from init, one stored round at a time. At the first stored round, the remaining clients' stored updates are applied unchanged. At each later one, those clients run calibration training for half the usual local steps, starting from the model rebuilt so far. Each new update keeps its own direction but is rescaled, per tensor, to the size of the stored update.
- **PGA** (Halimi et al. 2022): the reference model is the final model with client 0's share of its last FedAvg round removed. This is approximate, because client 0's last local model is taken to be `final + last_update`. Gradient ascent on client 0's labeled data follows, kept inside an L2 ball around the reference. The radius is ⅓ of the mean distance from the reference to random-init models. Ascent stops early once client 0's accuracy reaches chance. A few recovery FedAvg rounds without client 0 come last.

## Design notes

- **The pseudo-labeler is an explicit argument.** `local_train` takes the model that labels unlabeled data separately from the model being trained. Unlabeled data is the route by which a forgotten client's knowledge can come back during unlabeled-data training, so unlabeled-data training needs to be able to pick its pseudo-labeler. Options are the global model, the initial model, a retrained reference, or none. Per-class pseudo-label counts and precision are logged even when `lambda_u: 0`.
- **No DataLoaders.** The whole dataset is kept on the GPU, and augmentations run batched there. Each image gets its own RandAugment op and magnitude, using the ranges from the FixMatch paper. This avoids the cost of starting worker processes on Windows. Throughput is about 57 ms per local step with under 1 GB of VRAM.
- **BatchNorm momentum is 0.1, and the final layer starts at zero.** The pseudo-labeler runs in eval mode, so it relies on BatchNorm running stats. With FixMatch's 0.001 momentum, those stats lag far behind the weights. An early labeler then made confident labels that were all wrong (60% of samples above τ, 0% precision). With the zero-initialized output layer, an untrained labeler gives uniform probabilities and is never confident.
- **Plain PyTorch + torchvision.** Federated learning is simulated in one process by looping over clients.

## Reproducibility

Each run fixes the torch/numpy/python seeds. It saves the resolved config, the client sampling schedule, and the git commit. The commit is read from `.git/` directly, and uncommitted changes are not detected, so commit before runs that matter. Figures report mean ± std over 3 seeds.
