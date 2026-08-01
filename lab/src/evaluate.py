"""Evaluate a trained depth head on the held-out test split.

Usage:
    python -m src.evaluate
    python -m src.evaluate --checkpoint outputs/depth_head.pt
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import DepthDataset
from src.metrics import compute_all
from src.model import ViTDepthModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--data_dir", default=defaults.data_dir)
    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to <output_dir>/depth_head.pt")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    ckpt_path = args.checkpoint or os.path.join(args.output_dir, "depth_head.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ViTDepthModel(checkpoint["vision_model_name"]).to(device)
    model.head.load_state_dict(checkpoint["head_state_dict"])
    model.eval()

    test_ds = DepthDataset(args.data_dir, "test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    totals, n_batches = None, 0
    for pixel_values, depth_gt in test_loader:
        pixel_values, depth_gt = pixel_values.to(device), depth_gt.to(device)
        depth_pred = model(pixel_values)
        metrics = compute_all(depth_pred, depth_gt)

        if totals is None:
            totals = {k: 0.0 for k in metrics}
        for k, v in metrics.items():
            totals[k] += v
        n_batches += 1

    avg_metrics = {k: v / max(n_batches, 1) for k, v in (totals or {}).items()}

    print("Test set metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")

    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
