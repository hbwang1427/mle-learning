"""Run the trained depth model on a single image and save a visualization.

Usage:
    python -m src.infer --image data/synthetic_depth/images/test_0000.png
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.config import Config
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import ViTDepthModel


def colorize(depth_01):
    """Map a normalized [0, 1] depth map to an RGB image.

    Uses matplotlib's colormap if available (nicer for slides/reports),
    otherwise falls back to plain grayscale so this script has no hard
    dependency on matplotlib.
    """
    try:
        import matplotlib

        try:
            cmap = matplotlib.colormaps["plasma"]  # matplotlib >= 3.7
        except AttributeError:
            cmap = matplotlib.cm.get_cmap("plasma")  # older matplotlib
        colored = cmap(depth_01)[:, :, :3]
        return (colored * 255).astype(np.uint8)
    except ImportError:
        gray = (depth_01 * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", default=defaults.output_dir)
    parser.add_argument("--checkpoint", default=None,
                         help="Defaults to <output_dir>/depth_head.pt")
    parser.add_argument("--image_size", type=int, default=defaults.image_size)
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

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    image = Image.open(args.image).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)

    depth_pred = model(pixel_values).squeeze(0).cpu().numpy()  # (H, W) in [0, 1]

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    npy_path = os.path.join(args.output_dir, f"{base_name}_depth.npy")
    np.save(npy_path, depth_pred)

    vis = colorize(depth_pred)
    vis_path = os.path.join(args.output_dir, f"{base_name}_depth.png")
    Image.fromarray(vis).save(vis_path)

    print(f"Saved raw depth to {npy_path}")
    print(f"Saved colorized depth to {vis_path}")


if __name__ == "__main__":
    main()
