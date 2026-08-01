# DVC Practice: Versioning the Depth Dataset

This walks through the DVC workflow from Class 2 (MLE Tooling Practices)
using this lab's own synthetic depth dataset. Every command below has been
run end-to-end against this exact `lab/` folder, so you can follow it
verbatim.

Run everything from inside `lab/`.

## 0. Setup

```bash
pip install -r requirements.txt   # includes dvc
python data/generate_synthetic_depth.py
```

You should now have `data/synthetic_depth/{images,depth}/` plus
`train.jsonl` / `val.jsonl` / `test.jsonl`.

## 1. Initialize DVC

`lab/` is a subdirectory of the `mle-learning` git repo, so initialize DVC
here with `--subdir`:

```bash
dvc init --subdir
```

This creates `.dvc/` (config + cache pointers). It does **not** touch your
data yet.

## 2. Configure a remote

For practice, a local folder works fine as a "remote" -- it plays the same
role an S3/GCS bucket would in production, just without needing cloud
credentials.

```bash
mkdir -p ~/dvc-storage/mle-learning-lab
dvc remote add -d local_remote ~/dvc-storage/mle-learning-lab
```

## 3. Track the dataset

```bash
dvc add data/synthetic_depth
```

This creates `data/synthetic_depth.dvc` (a small pointer file, safe to
commit) and adds `data/synthetic_depth/` to `data/.gitignore` automatically
-- the actual images/depth arrays never go into git.

```bash
git add data/synthetic_depth.dvc data/.gitignore .dvc/config
git commit -m "data: track synthetic depth dataset v1 via DVC"
```

## 4. Push the data to your remote

```bash
dvc push
```

Check `~/dvc-storage/mle-learning-lab` -- your data bytes are now there,
content-addressed by hash, independent of git.

## 5. Prove the round trip: delete and restore

This is the actual point of DVC -- your data survives losing the local copy.

```bash
rm -rf data/synthetic_depth
ls data/                    # only the generator script is left
dvc pull
ls data/synthetic_depth/images | wc -l   # back to 320
```

## 6. Run the reproducible pipeline

`dvc.yaml` defines `train → evaluate` as a pipeline (data generation is left
as a manual step you already did in step 0, since that's what you just
practiced versioning by hand).

```bash
dvc repro
```

First run trains the model and evaluates it. Try running `dvc repro` again
immediately -- DVC detects nothing changed and skips everything:

```bash
dvc repro
# 'data/synthetic_depth.dvc' didn't change, skipping
# 'train' didn't change, skipping
# 'evaluate' didn't change, skipping
```

Now change a hyperparameter and re-run:

```bash
# edit params.yaml, e.g. bump train.epochs to 12
dvc repro
# only 'train' and 'evaluate' re-run -- DVC is a Makefile for ML
```

Visualize the pipeline and inspect metrics:

```bash
dvc dag
dvc metrics show
```

Commit the results (the metrics JSONs are small and meant for git; the
model checkpoint and dataset are DVC-cached, not git-tracked):

```bash
git add dvc.yaml dvc.lock outputs/train_metrics.json outputs/eval_metrics.json
git commit -m "train: baseline ViT depth head, AbsRel ~0.13"
dvc push
```

## 7. Lightweight experiment tracking with `dvc exp`

Try a couple of hyperparameter variants without committing each one as a
real branch:

```bash
dvc exp run --set-param train.lr=0.0005 --name lr-low
dvc exp run --set-param train.loss_fn=silog --name silog-loss
dvc exp show
```

`dvc exp show` prints a table comparing params and metrics across all runs.
If one wins, promote it:

```bash
dvc exp apply lr-low        # bring that run's results into your workspace
dvc exp branch lr-low exp/lr-low   # or turn it into a real git branch
```

## What you just practiced

| DVC Concept | Command | Matches Class 2 Section |
|---|---|---|
| Initialize DVC in a project | `dvc init` | "Setting Up DVC in a New Project" |
| Configure remote storage | `dvc remote add` | same |
| Track a dataset | `dvc add` | same |
| Push/pull data independent of git | `dvc push` / `dvc pull` | "DVC + Git Workflow" |
| Reproducible multi-stage pipeline | `dvc.yaml` + `dvc repro` | "DVC Pipelines" |
| Compare hyperparameter runs | `dvc exp run` / `dvc exp show` | "DVC Experiment Tracking" |

## Common gotchas

- **`dvc exp run` errors with "Git user name and email must be configured"**
  -- run `git config user.name "..."` / `git config user.email "..."` first
  (a one-time setup on any machine, not specific to this lab).
- **`dvc pull` says a target ".dvc file does not exist"** -- you're probably
  trying to pull a pipeline-stage output (like `outputs/depth_head.pt`)
  before ever running `dvc repro`. Run `dvc repro` at least once first.
- **Don't `dvc add` something that's already a pipeline `outs:` in
  `dvc.yaml`** -- pick one tracking mechanism per path. Here, the dataset is
  manually `dvc add`-ed and the pipeline stages just `deps:` on it; the
  model checkpoint is a pipeline `outs:` and is never manually `dvc add`-ed.
