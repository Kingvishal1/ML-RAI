#!/usr/bin/env python3
"""Tune the navigation (base), manipulation (arm) and selector policies with Optuna.

Optuna does not try every combination of hyperparameters. It runs one trial at a
time and uses the results so far to choose the next combination (TPE sampler),
and it stops trials that are clearly worse than earlier ones (median pruner).
All hyperparameters are searched together, so interactions between them are
explored.

Steps:
  1. Load a split dataset (dataset_base.npz / dataset_arm.npz from split_dataset.py,
     or a selector dataset from collect_selector_data.py) and split it 80/20 by
     trajectory, with the same split as the training scripts
  2. Run the current hand-picked configuration first as a baseline, then
     --trials Optuna trials
  3. Retrain the --top_k best configurations with --seeds random seeds each, so the
     final choice is not decided by one lucky seed
  4. Write a report (summary.md, trials.csv, best_params.json, plots)
  5. With --save_best, retrain the winner and save a checkpoint that
     run_policy_sim.py can load

Score that is minimised:
  base / arm : validation MSE in normalised units (same metric as the paper)
  selector   : validation masked cross-entropy (accuracy is recorded too)

One command for everything (all three policies at once, 2 parallel workers each):
  python experiments/optuna_tune.py --policy all --arm_data data/split/dataset_arm.npz \
      --base_data data/split/dataset_base.npz --selector_data data/dataset_selector.npz \
      --trials 100 --workers 2 --save_best
  Each policy gets its own subfolder and study. Its workers share the study (no
  repeated trials), a progress line is printed every minute, and when the search is
  done each policy's report and checkpoint are written. Leave out any --*_data to
  skip that policy. --workers also works with a single --policy.

Examples (run from final/robotnik_arm_task):
  python experiments/optuna_tune.py --policy arm  --data data/dataset_arm.npz  --trials 100
  python experiments/optuna_tune.py --policy base --data data/dataset_base.npz --trials 100 --max_samples 2M
  python experiments/optuna_tune.py --policy all --arm_data data/dataset_arm.npz \
      --base_data data/dataset_base.npz --info --max_samples 25%     (print data sizes only)

Data size (--max_samples for the search, --final_max_samples for --save_best):
  all | a row count: 500000, 500k, 2M | a share of the dataset: 25%, 0.25
  Rows are picked at random (the same rows every run), before the split by trajectory.
  python experiments/optuna_tune.py --policy selector --data data/dataset_selector.npz --trials 200
  python experiments/optuna_tune.py --policy arm --data data/dataset_arm.npz --trials 100 --save_best

Quick check that everything runs (a few minutes on a CPU):
  python experiments/optuna_tune.py --policy arm --data data/dataset_arm.npz --trials 6 --epochs 4 \
      --max_samples 20000 --top_k 2 --seeds 2

Results go to experiments/charts/optuna_<policy>_<date_time>/ (with --policy all, one
subfolder per policy). The study is stored in study.db there, so an interrupted run
can be resumed by rerunning the same command with --out_dir <that folder>.
--trials counts all trials per policy, including the baseline, split across workers.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import optuna
except ImportError:
    sys.exit("Optuna is not installed. Run:  pip install optuna")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from activations import ACTIVATIONS, get_activation  # noqa: E402
from regressors import build_mlp  # noqa: E402

# Number of candidate slots in the selector input (MAX_TARGETS in robotnik_arm_task.py).
# Kept here so this script does not need PyBullet or Pinocchio installed.
MAX_TARGETS = 5

# The configurations currently used by the trained policies. They are run as the
# first trial, so the report shows directly how they compare with Optuna's best.
BASELINES = {
    "regression": dict(num_layers=3, hidden_dim=512, activation="gelu", lr=1e-3,
                       batch_size=8192, loss="mae", optimizer="adamw", weight_decay=0.0,
                       dropout=0.0, scheduler="none", grad_clip=1.0),
    "selector": dict(num_layers=3, hidden_dim=128, activation="gelu", lr=1e-3,
                     batch_size="full", optimizer="adamw", weight_decay=1e-4, dropout=0.0),
}

REG_BATCH_CHOICES = [1024, 2048, 4096, 8192, 16384]
SEL_BATCH_CHOICES = [64, 256, 1024, "full"]


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def suggest_regression(trial, batch_choices):
    return dict(
        num_layers=trial.suggest_int("num_layers", 2, 5),
        hidden_dim=trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024]),
        activation=trial.suggest_categorical("activation", list(ACTIVATIONS)),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", batch_choices),
        loss=trial.suggest_categorical("loss", ["mse", "mae", "huber"]),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
        dropout=trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
        scheduler=trial.suggest_categorical("scheduler", ["none", "cosine", "plateau"]),
        grad_clip=trial.suggest_categorical("grad_clip", [0.0, 1.0, 5.0]),
    )


def suggest_selector(trial, batch_choices):
    return dict(
        num_layers=trial.suggest_int("num_layers", 1, 4),
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        activation=trial.suggest_categorical("activation", list(ACTIVATIONS)),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", batch_choices),
        optimizer=trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
        dropout=trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def trajectory_train_mask(traj_ids, val_frac=0.2, seed=42):
    """80/20 split by trajectory ID, identical to train_regressor.py / train_selector.py."""
    ids = np.unique(traj_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    train_ids = set(ids[:int(len(ids) * (1 - val_frac))].tolist())
    return np.array([t in train_ids for t in traj_ids])


def parse_size(text):
    """Parse a data size: 'all', a row count ('500000', '500k', '2M') or a share ('25%', '0.25').

    Returns None for all rows, an int for a row count, or a float in (0, 1) for a share.
    """
    if text is None:
        return None
    s = str(text).strip().lower().replace(",", "").replace("_", "")
    if s in ("all", "0", "none", ""):
        return None
    try:
        if s.endswith("%"):
            share = float(s[:-1]) / 100
        elif s.endswith(("k", "m")):
            return int(float(s[:-1]) * (1_000 if s.endswith("k") else 1_000_000))
        else:
            value = float(s)
            if value >= 1:
                return int(value)
            share = value
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid size '{text}': use all, a row count (500000, 500k, 2M) or a share (25%, 0.25)")
    if not 0 < share <= 1:
        raise argparse.ArgumentTypeError(f"invalid share '{text}': must be between 0% and 100%")
    return None if share == 1 else share


def subsample(arrays, size):
    """Keep a random subset of rows (same rows every run). size comes from parse_size."""
    n = len(arrays[0])
    keep = int(round(n * size)) if isinstance(size, float) else size
    if not keep or keep >= n:
        print(f"Using all {n:,} rows")
        return arrays
    idx = np.sort(np.random.RandomState(42).choice(n, keep, replace=False))
    print(f"Using {keep:,} of {n:,} rows ({100 * keep / n:.1f}%)")
    return [a[idx] for a in arrays]


def npz_header_shapes(path):
    """Array shapes in an .npz, read from the file headers without loading the data."""
    import zipfile
    shapes = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                        else np.lib.format.read_array_header_2_0)
                shapes[name[:-4]] = read(f)[0]
    return shapes


def print_data_info(jobs, size_text):
    """--info: print how big each dataset is and how many rows the search would use."""
    size = parse_size(size_text)
    print(f"--max_samples {size_text}:")
    for policy, path in jobs.items():
        shapes = npz_header_shapes(path)
        rows = shapes["inputs"][0]
        traj = len(np.unique(np.load(path)["metadata"][:, 0]))
        unit = "decisions" if policy == "selector" else "rows"
        out = shapes["labels"][1] if len(shapes["labels"]) > 1 else MAX_TARGETS
        used = rows if size is None else min(rows, int(round(rows * size)) if isinstance(size, float) else size)
        print(f"  {policy:8s} {path}\n"
              f"           {rows:,} {unit}, {shapes['inputs'][1]} inputs -> {out} outputs, "
              f"{traj} trajectories/episodes\n"
              f"           search would use {used:,} ({100 * used / rows:.1f}%)")


def load_regression_data(path, max_samples, features, device):
    d = np.load(path, allow_pickle=True)
    X, y, meta = d["inputs"].astype(np.float32), d["labels"].astype(np.float32), d["metadata"]
    names = [str(n) for n in d["feature_names"]] if "feature_names" in d.files else None
    if features:
        if names is None:
            sys.exit("--features needs a dataset with feature_names")
        missing = [f for f in features if f not in names]
        if missing:
            sys.exit(f"Features not in dataset: {missing}\nAvailable: {names}")
        X = X[:, [names.index(f) for f in features]]
        names = list(features)
    X, y, meta = subsample([X, y, meta], max_samples)

    tr = trajectory_train_mask(meta[:, 0])
    if tr.all() or not tr.any():
        sys.exit("Need at least two trajectories to make a train/validation split")
    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-8
    y_mean, y_std = y[tr].mean(0), y[tr].std(0) + 1e-8
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    print(f"Loaded {path}: {len(X)} rows, {X.shape[1]} inputs -> {y.shape[1]} outputs, "
          f"{len(np.unique(meta[:, 0]))} trajectories (train {tr.sum()} / val {(~tr).sum()} rows)")
    return dict(
        X_train=t((X[tr] - x_mean) / x_std), y_train=t((y[tr] - y_mean) / y_std),
        X_val=t((X[~tr] - x_mean) / x_std), y_val=t((y[~tr] - y_mean) / y_std),
        input_mean=x_mean, input_std=x_std, label_mean=y_mean, label_std=y_std,
        feature_names=names, in_dim=X.shape[1], out_dim=y.shape[1],
    )


def load_selector_data(path, max_samples, device):
    d = np.load(path, allow_pickle=True)
    X, labels, meta = d["inputs"].astype(np.float32), d["labels"].astype(np.int64), d["metadata"]
    X, labels, meta = subsample([X, labels, meta], max_samples)
    tr = trajectory_train_mask(meta[:, 0])
    if tr.all() or not tr.any():
        sys.exit("Need at least two episodes to make a train/validation split")
    # Standardise the positions only; the validity mask (last MAX_TARGETS columns) stays 0/1
    obs_mean = X[tr, :-MAX_TARGETS].mean(0)
    obs_std = X[tr, :-MAX_TARGETS].std(0) + 1e-8
    Xn = X.copy()
    Xn[:, :-MAX_TARGETS] = (X[:, :-MAX_TARGETS] - obs_mean) / obs_std
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    i = lambda a: torch.tensor(a, dtype=torch.long, device=device)  # noqa: E731
    print(f"Loaded {path}: {len(X)} decisions, {X.shape[1]} inputs, "
          f"{len(np.unique(meta[:, 0]))} episodes (train {tr.sum()} / val {(~tr).sum()})")
    return dict(
        X_train=f(Xn[tr]), y_train=i(labels[tr]), X_val=f(Xn[~tr]), y_val=i(labels[~tr]),
        input_mean=obs_mean, input_std=obs_std, in_dim=X.shape[1], out_dim=MAX_TARGETS,
    )


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------

def build_net(in_dim, out_dim, p):
    layers, prev = [], in_dim
    for _ in range(p["num_layers"]):
        layers += [nn.Linear(prev, p["hidden_dim"]), get_activation(p["activation"])]
        if p["dropout"] > 0:
            layers.append(nn.Dropout(p["dropout"]))
        prev = p["hidden_dim"]
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def plain_state_dict(net, in_dim, out_dim, p):
    """Copy weights into the build_mlp layout (no dropout layers) used by the sim loaders."""
    plain = build_mlp(in_dim, out_dim, p["hidden_dim"], p["num_layers"], p["activation"])
    src = [m for m in net if isinstance(m, nn.Linear)]
    dst = [m for m in plain if isinstance(m, nn.Linear)]
    for s, d in zip(src, dst):
        d.load_state_dict(s.state_dict())
    return plain.state_dict()


def masked_logits(logits, X):
    """Push invalid slots to a large negative score so softmax ignores them."""
    return logits.masked_fill(X[:, -MAX_TARGETS:] == 0, -1e9)


REG_LOSSES = {"mse": F.mse_loss, "mae": F.l1_loss, "huber": F.smooth_l1_loss}


@torch.no_grad()
def evaluate(net, X, y, selector, chunk=65536):
    net.eval()
    total, correct = 0.0, 0
    for i in range(0, len(X), chunk):
        xb, yb = X[i:i + chunk], y[i:i + chunk]
        out = net(xb)
        if selector:
            out = masked_logits(out, xb)
            total += F.cross_entropy(out, yb, reduction="sum").item()
            correct += int((out.argmax(1) == yb).sum())
        else:
            total += F.mse_loss(out, yb, reduction="sum").item() / yb.shape[1]
    n = len(X)
    return total / n, (100.0 * correct / n if selector else None)


def train(params, data, selector, epochs, patience, seed, device, trial=None):
    """Train one configuration. Returns the best validation score and the weights at that epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_net(data["in_dim"], data["out_dim"], params).to(device)
    opt_cls = torch.optim.AdamW if params["optimizer"] == "adamw" else torch.optim.Adam
    opt = opt_cls(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched_name = params.get("scheduler", "none")
    if sched_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif sched_name == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    else:
        sched = None
    loss_fn = None if selector else REG_LOSSES[params["loss"]]
    clip = params.get("grad_clip", 1.0)

    X, y = data["X_train"], data["y_train"]
    n = len(X)
    bs = n if params["batch_size"] == "full" else min(int(params["batch_size"]), n)
    best, best_acc, best_epoch, best_state, bad = math.inf, None, 0, None, 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = net(X[idx])
            if selector:
                loss = F.cross_entropy(masked_logits(out, X[idx]), y[idx])
            else:
                loss = loss_fn(out, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

        val, acc = evaluate(net, data["X_val"], data["y_val"], selector)
        if not math.isfinite(val):  # diverged (e.g. learning rate too high)
            if trial is not None:
                raise optuna.TrialPruned()
            break
        if sched_name == "plateau":
            sched.step(val)
        elif sched is not None:
            sched.step()

        if val < best:
            best, best_acc, best_epoch, bad = val, acc, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
        if trial is not None:
            trial.report(val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad >= patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    return dict(score=best, val_acc=best_acc, best_epoch=best_epoch, epochs_run=epoch, net=net)


@torch.no_grad()
def real_unit_mae(net, data):
    """Mean absolute error per output in physical units (m/s, rad/s or N m)."""
    net.eval()
    pred = torch.cat([net(data["X_val"][i:i + 65536]) for i in range(0, len(data["X_val"]), 65536)])
    err = (pred - data["y_val"]).abs().mean(0).cpu().numpy()
    return (err * data["label_std"]).tolist()


# ---------------------------------------------------------------------------
# Reporting and checkpoint
# ---------------------------------------------------------------------------

def fmt_params(p):
    return ", ".join(f"{k}={v:.2e}" if k == "lr" else f"{k}={v}" for k, v in p.items())


def write_trials_csv(study, path):
    """One row per trial (number, state, score, settings, extra info); no pandas needed."""
    import csv
    params = sorted({k for t in study.trials for k in t.params})
    attrs = sorted({k for t in study.trials for k in t.user_attrs})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "state", "value", *params, *attrs])
        for t in study.trials:
            w.writerow([t.number, t.state.name, t.value,
                        *[t.params.get(k, "") for k in params],
                        *[t.user_attrs.get(k, "") for k in attrs]])


def save_plots(study, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization import matplotlib as ovm
    except ImportError as e:
        print(f"Skipping plots ({e})")
        return
    plots = [("optimization_history.png", ovm.plot_optimization_history),
             ("param_importances.png", ovm.plot_param_importances),
             ("intermediate_values.png", ovm.plot_intermediate_values)]
    for name, fn in plots:
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax.flat[0].figure
            fig.set_size_inches(8, 5)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, name), dpi=150)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001  (plots are optional; e.g. too few trials)
            print(f"Skipping {name}: {e}")


def save_checkpoint(path, net, params, data, selector, result, epochs):
    sd = plain_state_dict(net.cpu(), data["in_dim"], data["out_dim"], params)
    ckpt = {
        "model_type": "selector_scene" if selector else "mlp",
        "model_state_dict": sd,
        "input_dim": data["in_dim"], "output_dim": data["out_dim"],
        "hidden_dim": params["hidden_dim"], "num_layers": params["num_layers"],
        "activation": params["activation"],
        "input_mean": data["input_mean"], "input_std": data["input_std"],
        "best_val_loss": result["score"], "best_epoch": result["best_epoch"],
        "epochs": epochs, "lr": params["lr"], "batch_size": params["batch_size"],
        "optuna_params": params,
    }
    if selector:
        ckpt["best_val_acc"] = result["val_acc"]
    else:
        ckpt.update(label_mean=data["label_mean"], label_std=data["label_std"],
                    action_dim=data["out_dim"], feature_names=data["feature_names"])
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Study helpers (shared by single runs, workers and the launcher)
# ---------------------------------------------------------------------------

POLICIES = ("base", "arm", "selector")


def batch_choices_for(policy, override):
    if override:
        return [b if b == "full" else int(b) for b in override]
    return SEL_BATCH_CHOICES if policy == "selector" else REG_BATCH_CHOICES


def storage_url(out_dir):
    return "sqlite:///" + os.path.abspath(os.path.join(out_dir, "study.db")).replace("\\", "/")


def open_study(policy, out_dir, seed, epochs):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna.create_study(
        study_name=f"{policy}_tuning", storage=storage_url(out_dir), load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=max(1, epochs // 10)),
    )


def enqueue_baseline(study, policy, batch_choices):
    """Queue the current configuration as the first trial. Returns True if queued."""
    baseline = BASELINES["selector" if policy == "selector" else "regression"]
    if baseline["batch_size"] not in batch_choices:
        print("Baseline batch size is not in --batch_choices; baseline trial skipped")
        return False
    study.enqueue_trial(baseline, user_attrs={"baseline": True})
    return True


def size_arg(text):
    """argparse type: validate a data size but keep the text, so it can be passed to workers."""
    parse_size(text)
    return str(text)


def collect_jobs(args):
    """{policy: dataset path} for this run, checking the files exist."""
    if args.policy == "all":
        jobs = {p: getattr(args, f"{p}_data") for p in POLICIES if getattr(args, f"{p}_data")}
        if not jobs:
            sys.exit("--policy all needs at least one of --base_data, --arm_data, --selector_data")
    else:
        if not args.data:
            sys.exit("--data is required")
        jobs = {args.policy: args.data}
    for p, path in jobs.items():
        if not os.path.exists(path):
            sys.exit(f"{p}: data file not found: {path}")
    return jobs


# ---------------------------------------------------------------------------
# Launcher: --policy all and/or --workers N
# ---------------------------------------------------------------------------

def launch(args):
    """Start worker processes for every requested policy, then write each final report.

    Each policy gets its own study (study.db). Its workers share that study, so they
    never repeat each other's trials and all learn from every result. When all
    workers are done, one extra run per policy re-runs the best configurations with
    several seeds, writes the report and (with --save_best) saves the checkpoint.
    """
    jobs = collect_jobs(args)
    root = args.out_dir or os.path.join(
        CHARTS_DIR, f"optuna_{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}")
    script = os.path.abspath(__file__)

    # Options passed through unchanged to every child process
    common = ["--max_samples", str(args.max_samples), "--device", args.device]
    for flag in ("epochs", "patience", "timeout"):
        if getattr(args, flag) is not None:
            common += [f"--{flag}", str(getattr(args, flag))]
    if args.batch_choices:
        common += ["--batch_choices", *args.batch_choices]

    def features_for(p):
        feats = args.features if args.policy != "all" else getattr(args, f"{p}_features", None)
        return ["--features", *feats] if feats and p != "selector" else []

    out_dirs, procs, studies = {}, [], {}
    per_worker = math.ceil(args.trials / args.workers)

    def start_worker(p, w):
        # -u: write log lines immediately instead of buffering them until the worker exits
        cmd = [sys.executable, "-u", script, "--policy", p, "--data", jobs[p], "--out_dir", out_dirs[p],
               "--trials", str(per_worker), "--seed", str(args.seed + w),
               "--search_only", "--no_baseline", *common, *features_for(p)]
        log = open(os.path.join(out_dirs[p], f"worker_{w}.log"), "w")
        procs.append((p, w, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))

    for p in jobs:
        out = os.path.join(root, p) if args.policy == "all" else root
        os.makedirs(out, exist_ok=True)
        out_dirs[p] = out
        # Create the study and queue the baseline once, before any worker starts
        studies[p] = open_study(p, out, args.seed, 1)
        if not args.no_baseline and len(studies[p].trials) == 0:
            enqueue_baseline(studies[p], p, batch_choices_for(p, args.batch_choices))
        start_worker(p, 0)

    # SQLite cannot stop two workers from taking the same queued (baseline) trial, so
    # the other workers start only once worker 0 has picked it up.
    waiting = optuna.trial.TrialState.WAITING
    deadline = time.time() + 600
    while time.time() < deadline and any(
            any(t.state == waiting for t in studies[p].get_trials(deepcopy=False))
            and procs[i][2].poll() is None for i, p in enumerate(jobs)):
        time.sleep(2)
    for p in jobs:
        for w in range(1, args.workers):
            start_worker(p, w)

    print(f"Started {len(procs)} worker process(es): {', '.join(jobs)} x {args.workers} worker(s), "
          f"~{per_worker} trials each")
    print(f"Output: {root}   (worker logs: <policy folder>/worker_N.log)")

    def status_line():
        parts = []
        for p in jobs:
            try:
                s = optuna.load_study(study_name=f"{p}_tuning", storage=storage_url(out_dirs[p]))
                done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
                pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in s.trials)
                best = f"{min(t.value for t in done):.6f}" if done else "-"
                parts.append(f"{p}: {len(done)} done, {pruned} pruned, best {best}")
            except Exception:  # noqa: BLE001  (study briefly locked or not created yet)
                parts.append(f"{p}: starting")
        return f"[{time.strftime('%H:%M:%S')}] " + " | ".join(parts)

    t_start = time.time()
    try:
        while any(pr.poll() is None for _, _, pr, _ in procs):
            time.sleep(args.poll)
            print(status_line(), flush=True)
    except KeyboardInterrupt:
        print("\nStopping workers ...")
        for _, _, pr, _ in procs:
            pr.terminate()
        for _, _, pr, log in procs:
            pr.wait()
            log.close()
        sys.exit("Stopped. Rerun the same command with --out_dir to resume.")
    for _, _, _, log in procs:
        log.close()
    print(status_line())
    print(f"Search finished in {(time.time() - t_start) / 60:.1f} min")

    failed = [(p, w) for p, w, pr, _ in procs if pr.returncode != 0]
    for p, w in failed:
        print(f"\nWorker {w} of {p} failed (exit code != 0). Last lines of its log:")
        with open(os.path.join(out_dirs[p], f"worker_{w}.log")) as f:
            print("".join(f.readlines()[-15:]))

    # --- Final step per policy: multi-seed re-run, report, optional checkpoint ---
    print("\nWriting final reports" + (" and checkpoints" if args.save_best else "") + " ...")
    finals = []
    for p, path in jobs.items():
        cmd = [sys.executable, "-u", script, "--policy", p, "--data", path, "--out_dir", out_dirs[p],
               "--trials", "0", "--no_baseline", "--top_k", str(args.top_k), "--seeds", str(args.seeds),
               "--seed", str(args.seed), *common, *features_for(p)]
        if args.save_best:
            cmd += ["--save_best", "--final_max_samples", str(args.final_max_samples)]
        log = open(os.path.join(out_dirs[p], "final.log"), "w")
        finals.append((p, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    for p, pr, log in finals:
        pr.wait()
        log.close()
        summary = os.path.join(out_dirs[p], "summary.md")
        if pr.returncode == 0 and os.path.exists(summary):
            with open(summary, encoding="utf-8") as f:
                result = [ln.strip() for ln in f if ln.startswith(("Best single", "Current config", "Selected"))]
            print(f"\n== {p} ==\n" + "\n".join(result) + f"\nReport: {summary}")
        else:
            print(f"\n== {p} == final step failed; see {os.path.join(out_dirs[p], 'final.log')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Optuna hyperparameter tuning for the BC policies",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--policy", required=True, choices=[*POLICIES, "all"],
                    help="Policy to tune, or 'all' to tune every policy given with --*_data at once")
    ap.add_argument("--data", default=None, help="Split dataset (.npz) for this policy")
    ap.add_argument("--base_data", default=None, help="With --policy all: base dataset")
    ap.add_argument("--arm_data", default=None, help="With --policy all: arm dataset")
    ap.add_argument("--selector_data", default=None, help="With --policy all: selector dataset")
    ap.add_argument("--base_features", nargs="+", default=None, help="With --policy all: like --features")
    ap.add_argument("--arm_features", nargs="+", default=None, help="With --policy all: like --features")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker processes per policy sharing one study (default 1)")
    ap.add_argument("--poll", type=float, default=60, help="Seconds between progress lines (launcher)")
    ap.add_argument("--search_only", action="store_true", help=argparse.SUPPRESS)  # set for workers
    ap.add_argument("--trials", type=int, default=100,
                    help="Optuna trials per policy, including the baseline (split across workers)")
    ap.add_argument("--timeout", type=float, default=None, help="Stop the search after this many hours")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Max epochs per trial (default: 40 for base/arm, 300 for selector)")
    ap.add_argument("--patience", type=int, default=None,
                    help="Early-stopping patience per trial (default: 10 for base/arm, 30 for selector)")
    ap.add_argument("--max_samples", type=size_arg, default="1M",
                    help="How much data the search uses: all, a row count (500000, 500k, 2M) "
                         "or a share (25%%, 0.25). Default 1M")
    ap.add_argument("--info", action="store_true",
                    help="Print how big each dataset is and how much --max_samples would use, then exit")
    ap.add_argument("--features", nargs="+", default=None,
                    help="Only use these input features (names from the dataset), e.g. to drop lidar")
    ap.add_argument("--batch_choices", nargs="+", default=None,
                    help="Override the batch sizes searched (use 'full' for full-batch)")
    ap.add_argument("--top_k", type=int, default=5,
                    help="Re-run this many best configurations with several seeds (0 = skip)")
    ap.add_argument("--seeds", type=int, default=3, help="Seeds per configuration in the re-run")
    ap.add_argument("--save_best", action="store_true",
                    help="Retrain the winner on --final_max_samples rows and save a checkpoint")
    ap.add_argument("--final_max_samples", type=size_arg, default="all",
                    help="Data for the --save_best retrain, same format as --max_samples. Default all")
    ap.add_argument("--no_baseline", action="store_true", help="Do not run the current configuration first")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--out_dir", default=None, help="Output folder; reuse an existing one to resume")
    args = ap.parse_args()

    if args.workers < 1:
        sys.exit("--workers must be at least 1")
    if args.info:
        print_data_info(collect_jobs(args), args.max_samples)
        return
    if args.policy == "all" or args.workers > 1:
        launch(args)
        return
    if not args.data:
        sys.exit("--data is required")

    selector = args.policy == "selector"
    epochs = args.epochs or (300 if selector else 40)
    patience = args.patience or (30 if selector else 10)
    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    batch_choices = batch_choices_for(args.policy, args.batch_choices)

    out_dir = args.out_dir or os.path.join(
        CHARTS_DIR, f"optuna_{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Policy: {args.policy}   device: {device}   output: {out_dir}")

    max_samples = parse_size(args.max_samples)
    if selector:
        data = load_selector_data(args.data, max_samples, device)
    else:
        data = load_regression_data(args.data, max_samples, args.features, device)

    # --- Study (stored in SQLite so it can be resumed or shared between processes) ---
    study = open_study(args.policy, out_dir, args.seed, epochs)
    suggest = suggest_selector if selector else suggest_regression
    enqueued_here = False
    if not args.no_baseline and len(study.trials) == 0:
        enqueued_here = enqueue_baseline(study, args.policy, batch_choices)

    def objective(trial):
        params = suggest(trial, batch_choices)
        t0 = time.time()
        res = train(params, data, selector, epochs, patience, args.seed, device, trial)
        trial.set_user_attr("best_epoch", res["best_epoch"])
        trial.set_user_attr("epochs_run", res["epochs_run"])
        trial.set_user_attr("minutes", round((time.time() - t0) / 60, 2))
        if selector:
            trial.set_user_attr("val_acc", res["val_acc"])
        return res["score"]

    def log_trial(study_, trial):
        tag = " (baseline)" if trial.user_attrs.get("baseline") else ""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            extra = f"  acc={trial.user_attrs['val_acc']:.1f}%" if selector else ""
            best = study_.best_value
            print(f"[trial {trial.number:3d}]{tag} score={trial.value:.6f}{extra}  "
                  f"best={best:.6f}  ({trial.user_attrs.get('minutes', 0)} min)")
        else:
            last = max(trial.intermediate_values) if trial.intermediate_values else 0
            print(f"[trial {trial.number:3d}]{tag} {trial.state.name.lower()} at epoch {last}")

    if enqueued_here:
        print("Trial 0 is the current configuration (baseline)")
    t_start = time.time()
    try:
        study.optimize(objective, n_trials=args.trials,
                       timeout=args.timeout * 3600 if args.timeout else None,
                       callbacks=[log_trial], gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nSearch interrupted; reporting on the trials finished so far.")
    search_min = (time.time() - t_start) / 60
    if args.search_only:  # worker process: the launcher writes the report afterwards
        print(f"Worker finished in {search_min:.1f} min")
        return

    states = optuna.trial.TrialState
    complete = [t for t in study.trials if t.state == states.COMPLETE]
    if not complete:
        sys.exit("No trial completed; nothing to report.")
    complete.sort(key=lambda t: t.value)
    base_trial = next((t for t in study.trials if t.user_attrs.get("baseline")), None)

    # --- Re-run the best configurations with several seeds ---
    reeval = []
    if args.top_k > 0 and args.seeds > 0:
        candidates = complete[:args.top_k]
        if base_trial is not None and base_trial.state == states.COMPLETE and base_trial not in candidates:
            candidates.append(base_trial)
        print(f"\nRe-running {len(candidates)} configurations x {args.seeds} seeds ...")
        for t in candidates:
            scores, accs = [], []
            for s in range(args.seeds):
                r = train(t.params, data, selector, epochs, patience, args.seed + 1 + s, device)
                scores.append(r["score"])
                accs.append(r["val_acc"])
            row = dict(trial=t.number, baseline=bool(t.user_attrs.get("baseline")),
                       mean=float(np.mean(scores)), std=float(np.std(scores)), params=t.params)
            if selector:
                row["acc_mean"] = float(np.mean(accs))
            reeval.append(row)
            print(f"  trial {t.number:3d}: {row['mean']:.6f} +/- {row['std']:.6f}"
                  + ("  (baseline)" if row["baseline"] else ""))
        reeval.sort(key=lambda r: r["mean"])
        winner = reeval[0]["params"]
    else:
        winner = complete[0].params

    # --- Report files ---
    write_trials_csv(study, os.path.join(out_dir, "trials.csv"))
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump({"policy": args.policy, "best_params": winner, "reeval": reeval}, f, indent=2)
    save_plots(study, out_dir)

    importances = {}
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:  # noqa: BLE001
        print(f"Parameter importance not available: {e}")

    metric = "validation masked cross-entropy" if selector else "validation MSE (normalised)"
    pruned = sum(t.state == states.PRUNED for t in study.trials)
    lines = [
        f"# Optuna tuning: {args.policy} policy", "",
        f"- Data: `{args.data}` ({len(data['X_train'])} train / {len(data['X_val'])} val rows, "
        f"split by trajectory)",
        f"- Trials: {len(study.trials)} ({len(complete)} complete, {pruned} pruned), "
        f"search time {search_min:.1f} min on {device}",
        f"- Sampler: TPE; pruner: median; max {epochs} epochs per trial, patience {patience}",
        f"- Score minimised: {metric}", "",
        "## Result", "",
        f"Best single trial: **{complete[0].value:.6f}** (trial {complete[0].number})",
    ]
    if base_trial is not None and base_trial.state == states.COMPLETE:
        change = 100 * (complete[0].value - base_trial.value) / base_trial.value
        lines.append(f"Current configuration (baseline): **{base_trial.value:.6f}** "
                     f"-> best is {change:+.1f}% relative to it")
    lines += ["", f"Selected configuration: `{fmt_params(winner)}`", ""]
    if reeval:
        lines += [f"## Top configurations re-run with {args.seeds} seeds", "",
                  "| Trial | Mean | Std |" + (" Acc % |" if selector else "") + " Settings |",
                  "|---|---|---|" + ("---|" if selector else "") + "---|"]
        for r in reeval:
            name = f"{r['trial']}" + (" (baseline)" if r["baseline"] else "")
            acc = f" {r['acc_mean']:.1f} |" if selector else ""
            lines.append(f"| {name} | {r['mean']:.6f} | {r['std']:.6f} |{acc} {fmt_params(r['params'])} |")
        lines += ["", "If two configurations differ by less than about one standard deviation, "
                      "treat them as equally good.", ""]
    if importances:
        lines += ["## Parameter importance (fANOVA)", "", "| Parameter | Importance |", "|---|---|"]
        lines += [f"| {k} | {v:.3f} |" for k, v in importances.items()]
        lines.append("")
    lines += ["## Top 10 trials", "", "| Trial | Score | Best epoch | Settings |", "|---|---|---|---|"]
    for t in complete[:10]:
        lines.append(f"| {t.number} | {t.value:.6f} | {t.user_attrs.get('best_epoch')} | {fmt_params(t.params)} |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[:14]))

    # --- Optional: retrain the winner and save a checkpoint ---
    if args.save_best:
        print("\nRetraining the selected configuration for the checkpoint ...")
        final_rows = parse_size(args.final_max_samples)
        if final_rows != max_samples:
            if selector:
                data = load_selector_data(args.data, final_rows, device)
            else:
                data = load_regression_data(args.data, final_rows, args.features, device)
        res = train(winner, data, selector, epochs, patience, args.seed, device)
        name = f"optuna_{args.policy}_{winner['activation']}_h{winner['hidden_dim']}_l{winner['num_layers']}.pt"
        if not selector:
            mae = real_unit_mae(res["net"], data)
            print("Validation MAE per output (real units): " + ", ".join(f"{m:.4f}" for m in mae))
        save_checkpoint(os.path.join(out_dir, name), res["net"], winner, data, selector, res, epochs)

    print(f"\nDone. Report: {os.path.join(out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
