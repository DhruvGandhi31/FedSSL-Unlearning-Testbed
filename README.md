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
| 4. Unlearning | retrain, FedEraser, projected gradient ascent | built and run once; FedEraser BN stats and PGA stop rule fixed (see [Results](#results)) |
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
analysis/     # reads results/ only; summarize.py prints the results tables
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
- **FedEraser** (Liu et al. 2021): rebuilds the model from init, one stored round at a time. At the first stored round, the remaining clients' stored updates are applied unchanged. At each later one, those clients run calibration training for half the usual local steps, starting from the model rebuilt so far. Each new update keeps its own direction but is rescaled, per tensor, to the size of the stored update. Only learnable parameters are rebuilt this way. After each step, the BatchNorm running statistics are reset and re-estimated with a forward pass over 5,000 images from that step's remaining clients. Rescaled statistic deltas don't follow the new weights; see the results below for what happened without this.
- **PGA** (Halimi et al. 2022): the reference model is the final model with client 0's share of its last FedAvg round removed. This is approximate, because client 0's last local model is taken to be `final + last_update`. Gradient ascent on client 0's labeled data follows (lr 1e-4), kept inside an L2 ball around the reference. The radius is ⅓ of the mean distance from the reference to random-init models. Ascent stops once client 0's accuracy is no higher than the *reference model's* test accuracy reweighted to client 0's class mix (`pga_stop: matched`). That is, the model should do no better on client 0's data than on unseen data with the same classes. The target is fixed at the reference: computed from the current model, it falls as ascent damages the model. A few recovery FedAvg rounds without client 0 come last.

## Results

Single seed (0) so far, not yet mean ± std over 3 seeds. To regenerate the tables: `python -m analysis.summarize results/excl_class_ssl/<run> --diagnose`.

### Original SSL run: `configs/excl_class_ssl.yaml`

Run `results/excl_class_ssl/s0-20261003-001110`, commit `b9d1980`. 200 rounds took 56.8 min (17.0 s/round).

| metric | value |
|---|---|
| final test accuracy | 0.592 (best 0.641, at round 189) |
| **class-0 (forget class) test accuracy** | **0.000 at the end, and at or near 0 throughout** (max 0.006, at round 64) |
| per-class test accuracy, classes 0–9 | 0.00, 0.97, 0.85, 0.37, 0.65, 0.14, 0.85, 0.57, 0.83, 0.69 |
| confident pseudo-labels, all unlabeled pools | 16,152 of 45,405 (35.6%), precision 0.845 |
| confident class-0 pseudo-labels on clients 1–9 | 0 |

Class-0 accuracy at every 25th round: r24 0.000, r49 0.002, r74 0.000, r99 0.000, r124 0.000, r149 0.000, r174 0.000, r199 0.000.

### Finding: the global model never learns the forget class

With the defaults, client 0 holds the only 100 labeled class-0 images. Under that setup, class 0 never makes it into the global model, so in this run there is nothing for unlearning to remove.

- **Client 0 does learn class 0 locally.** One round of client 0's local training, starting from the final model, takes class-0 accuracy from 0.00 to 0.77. The same round wipes out classes client 0 doesn't hold: classes 1, 5 and 8 drop to 0.00. This is ordinary non-IID client drift.
- **Averaging erases it every round.** Client 0 is 1 of 5 participants, and is in 86 of 200 rounds. The other 4 clients never see a class-0 label. After averaging, class-0 test images are predicted mostly as ship (391/1000) and bird (347/1000).
- **SSL actively pushes class 0 down.** Clients 1–9 hold 4,000 unlabeled class-0 images. The final model labels 595 of them (14.9%) confidently, and all of those labels are wrong: 429 bird, 130 ship. This is the reverse of the propagation the project is looking for. The supervised control (`lambda_u: 0`) has no such pressure, so class 0 might be learned there but not under SSL. That run would confirm or rule this out.

**Fixes tried.** These are 30-round probes at constant lr, which is close to the first 30 rounds of the 200-round cosine schedule; the baseline was 0.000 at rounds 24 and 29.

| probe | class-0 acc by round 30 | verdict |
|---|---|---|
| SSL, all of client 0's class-0 images labeled (`forget_labeled_frac: 1.0`, 1,000 labels, 54% of its labeled set) | 0.000 (blip of 0.039 at round 4) | label count isn't the lever |
| supervised, same | 0.000 | **SSL pseudo-labels aren't the main cause**: plain FedAvg erases it too |
| SSL, same labels, `local_steps: 10` × 150 rounds (same compute) | 0.088 / **0.436** / 0.000 / 0.111 / 0.158 / 0.001 at rounds 74 / 89 / 104 / 119 / 134 / 149 | appears, but only in rounds client 0 joined |
| supervised, same | 0.000 / **0.308** / 0.000 / 0.075 / 0.022 / 0.000 at the same rounds | same pattern |

- **With 10 local steps, class 0 is nonzero only at evaluations where client 0 took part in that same round.** One round without client 0 takes it back to 0, in both SSL and supervised runs. Zero confident class-0 pseudo-labels ever appear on clients 1–9.
- **So the global model never keeps class-0 knowledge.** It reflects only client 0's most recent update.
- **Fewer local steps reduce drift enough for class 0 to show up, but not enough for it to persist.** And since one FedAvg round without client 0 erases it, any unlearning method that runs even one such round would "forget" it trivially.

- **Not a bias problem.** On the original final model, class 0's output bias is normal (−0.05 vs −0.13…0.31 for the others). Its output-weight norm is the smallest (1.78 vs 2.1–2.7).
- **Class 0 isn't recognized at all.** On class-0 test images, class 0's score sits at a near-random position among the 10 classes and is never top. Adding to the bias doesn't recover it: +4 gives 0.29, and +8 predicts class 0 for almost everything.
- **So the averaged model never represents class 0.** Client 0's 50 local steps learn it, but averaging with four clients that drift away from it erases it every round.

**Consequence:** with the current scenario, every method's class-0 gap against retrain will be about 0 for trivial reasons. The scenario needs changing before the main figure means anything. Options, not yet decided:

1. Give client 0 more class-0 signal: raise `forget_share` and/or label more of client 0's class-0 images.
2. Weight FedAvg by labeled-data size or by class coverage, rather than equally.
3. Lower τ, or warm up λ_u, so early wrong pseudo-labels don't lock class 0 out.
4. Run `configs/excl_class_sup.yaml` first, to see whether class 0 survives without SSL.

### Unlearning on this run

| model | test acc | class-0 acc | class-0 gap vs retrain | confident class-0 PLs on clients 1–9 |
|---|---|---|---|---|
| original | 0.5920 | 0.0000 | +0.0000 | 0 |
| retrain | 0.5534 | 0.0000 | — | 0 |
| FedEraser, `--labeler current` (before BN fix) | 0.3074 | 0.0000 | +0.0000 | 0 |
| FedEraser, `--labeler original` (before BN fix) | 0.3326 | 0.0000 | +0.0000 | 0 |
| FedEraser, `--labeler current` (BN fix) | 0.2002 | 0.0000 | +0.0000 | 0 |
| FedEraser, `--labeler original` (BN fix) | 0.4085 | 0.0000 | +0.0000 | 0 |
| PGA, `--labeler current` (old settings) | 0.4141 | 0.0000 | +0.0000 | 0 |
| **PGA, `--labeler current` (fixed)** | **0.5492** | 0.0000 | +0.0000 | 0 |

Every gap is zero, but trivially so: the original model never learned class 0 (see the finding above). These numbers say nothing yet about the research question. What they do show is how each implementation behaves on a real run. Two implementation problems turned up and have since been fixed; the scenario problem remains.

- **Retrain** (`unlearn/retrain-20261003-010808`) took 112.3 min, about 34 s/round against 17 s/round for the original. The GPU was mostly idle, so the slowdown was probably CPU scheduling while the machine sat unattended; nothing else was using the CPU. Final per-class accuracy: 0.00, 0.97, 0.89, 0.16, 0.56, 0.07, 0.84, 0.57, 0.92, 0.56. Class 0 was 0.000 at every eval round. Original and retrain agree on class 0, so with this scenario the forget metric can't tell them apart.
- **FedEraser** (`federaser-current-20261003-030038`, `federaser-original-20261003-031530`) took 14.7 and 14.6 min, 40 calibration steps each. Utility is low: test accuracy is 0.31 and 0.33, against 0.55 for retrain.
  - **The reconstruction is unstable.** At some steps the model predicts class 0 for every test image (test accuracy exactly 0.10): steps 5 and 19 with `current`, steps 5, 15 and 19 with `original`. It then recovers.
  - **Likely cause: BatchNorm running statistics.** The calibration rescaled their deltas like weight deltas, which was my implementation choice and not something the paper specifies. FedEraser's final model has median `running_var` 1.01, close to its initial value of 1. The original and retrain models have 0.15. Stored running-variance deltas are also large compared with weight deltas (norm 12.6 vs 1.7 in a round-195 update). Running statistics are now left out of calibration and re-estimated on the rebuilt model at each step (see Unlearning methods), and `bn_var_median` is logged at every step.
  - **After the BN fix** (`federaser-current-20261003-045421`, `federaser-original-20261003-050936`; 15.1 and 21.2 min):
    - The collapses are gone at every step.
    - Test accuracy is 0.200 with `current` (was 0.307) and 0.409 with `original` (was 0.333).
  - **The diagnosis above was wrong.** Re-estimated from real data, the running variance is still high (median 0.95–1.7). So the 1.01 was the true activation scale of the rebuilt weights, not stale statistics. The real limit is the rebuilt weights themselves:
    - The conv weight norm is 51 for both FedEraser versions, against 30 for the original and for retrain (60 at init).
    - Normal training halves it through 10k steps of weight decay. FedEraser replays 1 stored round per 5-round interval, so it applies only about a fifth of that trajectory and the weights stay too large.
    - This comes from FedEraser's interval approximation at `history_interval: 5`, not from the BN handling.
    - Options: `history_interval: 1`, which costs about 3 GB of history and about 75 min per FedEraser run; scaling each calibrated update by the interval length, which is not in the paper; or accepting the lower utility as a known property of FedEraser.
  - **The labeler effect got bigger.** With correct statistics, the half-rebuilt `current` labeler is confident on 31.5% of samples, but only 30.3% of those labels are right. That noise drags FedEraser down to 0.200. The `original` labeler (the contaminated final model) stays at 0.87 precision and gives 0.409. So FedEraser's utility on SSL data depends heavily on which model pseudo-labels during calibration. That is exactly the lever the project's hypothesis is about.
  - **The labeler matters.** During calibration, the `original` labeler (the contaminated final model) produced confident pseudo-labels on 34.3% of unlabeled samples with 0.871 precision. The `current` labeler (the half-rebuilt model) managed only 4.6%, with 0.603 precision. With `original` the result was slightly better (0.333 vs 0.307). Neither produced any class-0 pseudo-labels, because the original model has no class-0 knowledge to pass on. This is the mechanism the project wants to measure, and it is active here. It just has nothing to carry in this scenario.
- **PGA, old settings** (`pga-current-20261003-033014`, lr 0.01, stop at chance accuracy, checked every 10 steps). Took 4.0 min.
  - The reference model already scores 0.488 on client 0's labeled data, which matches retrain's 0.490, and 0.550 test accuracy.
  - **The ascent wrecked the model anyway.** By the first check, at step 10, the loss was 72.5 and test accuracy had collapsed to 0.10.
  - **Recovery didn't fully repair it.** Five recovery rounds brought test accuracy back to 0.414.
- **What the PGA sweep showed** (ascent only, lr 1e-2 to 3e-4, radius ⅓ to 1/100 of the distance to random-init models):
  - **The ball radius doesn't matter.** Every setting hit chance accuracy within 1–2 steps, after moving only 0.05–0.74 from the reference. Even the smallest radius (0.72) never bound.
  - **The stop condition was the real problem.** Driving client 0 to chance accuracy over-unlearns: retrain itself scores 0.49 on client 0's data, and getting to 0.10 means destroying shared features. One step at lr 1e-2 takes test accuracy to 0.13.
- **PGA, fixed** (`pga-current-20261003-044724`): lr 1e-4, stop condition checked every step, `pga_stop: matched` with the target fixed at the reference (0.445).
  - Ascent stopped after 1 step, with client 0's accuracy at 0.415.
  - Test accuracy was 0.527 just after the ascent and **0.549** after recovery, against 0.553 for retrain.
  - An intermediate attempt (`pga-current-20261003-044233`) recomputed the target from the current model at every step. That target fell from 0.445 to 0.024 as the model degraded, so the ascent ran 8 steps and collapsed the model before stopping (final 0.536).
- `unlearn/federaser-current-20261003-030102` is an aborted duplicate. A second launcher script accidentally started the same job 24 s after the real one and was stopped. It has no `final.pt`, so `analysis/summarize.py` ignores it. It can be deleted.

## Design notes

- **The pseudo-labeler is an explicit argument.** `local_train` takes the model that labels unlabeled data separately from the model being trained. Unlabeled data is the route by which a forgotten client's knowledge can come back during unlabeled-data training, so unlabeled-data training needs to be able to pick its pseudo-labeler. Options are the global model, the initial model, a retrained reference, or none. Per-class pseudo-label counts and precision are logged even when `lambda_u: 0`.
- **No DataLoaders.** The whole dataset is kept on the GPU, and augmentations run batched there. Each image gets its own RandAugment op and magnitude, using the ranges from the FixMatch paper. This avoids the cost of starting worker processes on Windows. Throughput is about 57 ms per local step with under 1 GB of VRAM.
- **BatchNorm momentum is 0.1, and the final layer starts at zero.** The pseudo-labeler runs in eval mode, so it relies on BatchNorm running stats. With FixMatch's 0.001 momentum, those stats lag far behind the weights. An early labeler then made confident labels that were all wrong (60% of samples above τ, 0% precision). With the zero-initialized output layer, an untrained labeler gives uniform probabilities and is never confident.
- **Plain PyTorch + torchvision.** Federated learning is simulated in one process by looping over clients.

## Reproducibility

Each run fixes the torch/numpy/python seeds. It saves the resolved config, the client sampling schedule, and the git commit. The commit is read from `.git/` directly, and uncommitted changes are not detected, so commit before runs that matter. Figures report mean ± std over 3 seeds.
