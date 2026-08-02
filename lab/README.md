# Lab: ViT Depth Estimation + DVC Practice
##test
A hands-on companion to Class 1 (Course Project Preview: Monocular Depth
Estimation) and Class 2 (MLE Tooling Practices). You'll fine-tune a small
head on top of a **frozen Vision Transformer** (Hugging Face `transformers`)
to predict depth maps from RGB images, and use this project's own dataset to
practice **DVC** (data version control) end to end.

## What's here

```
lab/
  data/
    generate_synthetic_depth.py   # builds a tiny offline RGB -> depth dataset
  src/
    config.py                     # hyperparameters / model name
    model.py                      # ViT encoder (frozen) + depth head (trainable)
    dataset.py                    # manifest -> (pixel_values, depth) dataset
    losses.py                     # L1 and scale-invariant (Eigen et al.) losses
    metrics.py                    # AbsRel, RMSE, delta<1.25
    train.py                      # training loop
    evaluate.py                   # test-set evaluation
    infer.py                      # run the model on a single image
  params.yaml                     # hyperparameters DVC watches for changes
  dvc.yaml                        # train -> evaluate pipeline
  requirements.txt
  DVC_PRACTICE.md                 # step-by-step DVC exercise using this data
```

## Why this architecture

The model follows the same pattern used throughout this course (see
`ml-multimodal`'s `FusionCLIPModel`): a **frozen pretrained backbone** plus a
**small trainable head**.

```
image ──► ViT encoder (frozen, e.g. WinKawaks/vit-tiny-patch16-224)
              │ patch tokens (drop CLS), reshaped to a feature map
              ▼
          Conv depth head (trainable) ──► sigmoid ──► bilinear upsample ──► depth map
```

This is a minimal version of DPT (Dense Prediction Transformer) from the
Class 1 notes: ViT encoder → reassemble tokens into a feature map → a small
decoder → depth. Only the head trains (~5% of total parameters here), so
training is cheap and fast even on a laptop CPU.

**Swapping the backbone**: change `vision_model_name` in `src/config.py` or
`params.yaml` to any HF `AutoModel` that exposes a ViT-style
`last_hidden_state` (e.g. `google/vit-base-patch16-224-in21k` for a full-size
ViT, or `facebook/dinov2-small`).

## Setup

```bash
cd lab
pip install -r requirements.txt
```

## Running it

```bash
# 1. Generate the synthetic dataset (240 train / 40 val / 40 test)
python data/generate_synthetic_depth.py

# 2. Train the depth head
python -m src.train

# 3. Evaluate on the held-out test split
python -m src.evaluate

# 4. Run inference on a single image
python -m src.infer --image data/synthetic_depth/images/test_0000.png
```

`train.py` prints the trainable-vs-total parameter count each run (typically
~5% of total, matching the "light training" pattern from the rest of this
course), plus per-epoch train/val loss and AbsRel / delta<1.25 metrics.
Override any hyperparameter via flags:

```bash
python -m src.train --epochs 15 --batch_size 8 --lr 3e-4 --loss_fn silog
```

On the default settings, expect val AbsRel to drop from ~0.28 to ~0.13 and
delta<1.25 to climb to ~0.92 within 8 epochs (a few minutes on CPU) --
confirming the model is actually learning the synthetic task, not just
running.

## The dataset

`data/generate_synthetic_depth.py` procedurally draws a scene per sample: a
background that recedes into the distance (far at the top, near at the
bottom -- a toy ground plane) with 1-3 colored shapes placed on top, each
sitting closer to the camera than the background behind it. The model has to
learn "shape presence + position + size implies foreground depth" from RGB
alone -- a small, fully offline stand-in for real monocular depth datasets
like NYU-Depth-V2 or KITTI.

Manifest format (`train.jsonl` / `val.jsonl` / `test.jsonl`), one JSON object
per line:

```json
{"image": "train_0000.png", "depth": "train_0000.npy", "prompt": "a red circle and an orange circle"}
```

Depth is stored as a float32 `.npy` array normalized to `[0, 1]` (0 = near,
1 = far). Real datasets typically store depth as 16-bit PNG; `.npy` keeps
this lab dependency-free. The `prompt` field is unused by this lab's task
but is there so you can later try the "text-guided depth" extension
mentioned in Class 1 (depth + RGB + text).

**Using a real dataset instead**: point `src/dataset.py` at any
`{"image", "depth"}` manifest format, e.g. built from NYU-Depth-V2 or a
Hugging Face Hub depth dataset -- no other code changes needed.

## Practicing DVC

This lab's dataset is deliberately generated locally rather than committed
to git -- that's the point. See **`DVC_PRACTICE.md`** for the full walkthrough:
`dvc init`, `dvc add`, a local remote, `dvc push`/`pull`, `dvc repro` on the
`train → evaluate` pipeline in `dvc.yaml`, and `dvc exp run` for lightweight
experiment tracking.

## Extending

- **Loss function**: try `--loss_fn silog` (scale-invariant log loss) vs the
  default `l1` and compare `outputs/eval_metrics.json`.
- **Unfreeze the backbone**: set `freeze_backbone=False` in `ViTDepthModel`
  for full fine-tuning once the frozen-backbone baseline works (more
  compute, likely better accuracy).
- **Bigger backbone**: swap in `google/vit-base-patch16-224-in21k`.
- **Real data**: swap the synthetic generator for NYU-Depth-V2 / KITTI and
  re-run the same pipeline.
