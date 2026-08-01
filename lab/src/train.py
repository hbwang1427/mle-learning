"""Train the depth head on top of a frozen ViT backbone.

Usage:
    python -m src.train
    python -m src.train --epochs 10 --batch_size 8 --lr 3e-4 --loss_fn silog
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import DepthDataset
from src.losses import get_loss_fn
from src.metrics import compute_all
from src.model import ViTDepthModel


def load_params(path="params.yaml"):
    """Read the `train:` section of params.yaml, if present.

    This is what makes `dvc repro` meaningful: DVC watches params.yaml for
    changes and re-runs this stage when it changes, but DVC itself never
    injects values into the script -- the script has to read the file.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("train", {})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    params = load_params()  # params.yaml overrides Config; CLI flags override both

    parser.add_argument("--vision_model_name",
                         default=params.get("vision_model_name", defaults.vision_model_name))
    parser.add_argument("--data_dir", default=defaults.data_dir)
    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--loss_fn", default=params.get("loss_fn", defaults.loss_fn),
                         choices=["l1", "silog"])
    parser.add_argument("--lr", type=float, default=params.get("lr", defaults.lr))
    parser.add_argument("--batch_size", type=int,
                         default=params.get("batch_size", defaults.batch_size))
    parser.add_argument("--epochs", type=int, default=params.get("epochs", defaults.epochs))
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=params.get("seed", defaults.seed))
    return parser.parse_args()


@torch.no_grad()
def evaluate_loader(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_metrics, n_batches = 0.0, None, 0
    for pixel_values, depth_gt in loader:
        pixel_values, depth_gt = pixel_values.to(device), depth_gt.to(device)
        depth_pred = model(pixel_values)
        loss = loss_fn(depth_pred, depth_gt)
        metrics = compute_all(depth_pred, depth_gt)

        total_loss += loss.item()
        if total_metrics is None:
            total_metrics = {k: 0.0 for k in metrics}
        for k, v in metrics.items():
            total_metrics[k] += v
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {k: v / max(n_batches, 1) for k, v in (total_metrics or {}).items()}
    return avg_loss, avg_metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DepthDataset(args.data_dir, "train")
    val_ds = DepthDataset(args.data_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = ViTDepthModel(args.vision_model_name).to(device)
    trainable, total = model.param_counts()
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}% of total)")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    loss_fn = get_loss_fn(args.loss_fn)

    history = []
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.output_dir, "depth_head.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running_loss = 0.0

        for pixel_values, depth_gt in train_loader:
            pixel_values, depth_gt = pixel_values.to(device), depth_gt.to(device)

            optimizer.zero_grad()
            depth_pred = model(pixel_values)
            loss = loss_fn(depth_pred, depth_gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_metrics = evaluate_loader(model, val_loader, loss_fn, device)
        elapsed = time.time() - start

        print(f"epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_abs_rel={val_metrics['abs_rel']:.4f}  "
              f"val_delta1.25={val_metrics['delta1_25']:.4f}  "
              f"({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Only the small trainable head needs to be saved -- the frozen
            # backbone is re-downloaded/re-loaded from its HF checkpoint id.
            torch.save({
                "head_state_dict": model.head.state_dict(),
                "vision_model_name": args.vision_model_name,
            }, ckpt_path)

    metrics_path = os.path.join(args.output_dir, "train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "history": history,
            "best_val_loss": best_val_loss,
            "final_val_metrics": history[-1] if history else {},
        }, f, indent=2)

    print(f"Saved best head checkpoint to {ckpt_path}")
    print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
