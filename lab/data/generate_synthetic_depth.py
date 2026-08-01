"""Generate a tiny, fully offline RGB -> depth dataset for the ViT depth lab.

Each sample is a procedurally drawn scene: a background that recedes into the
distance (far at the top, near at the bottom, like a ground plane) with a
handful of colored shapes placed on top. Each shape sits closer to the camera
than the local background behind it, so the model has to learn "shape
presence + position + size implies foreground depth" from RGB alone -- a
toy version of real monocular depth estimation.

Depth values are normalized to [0, 1] (0 = near, 1 = far) and saved as
float32 .npy arrays. Real datasets like NYU-Depth-V2 / KITTI typically store
depth as 16-bit PNG; we use .npy here to keep the lab dependency-free and the
values easy to inspect with plain numpy.

Run:
    python data/generate_synthetic_depth.py
"""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

SHAPE_COLORS = {
    "red": (220, 60, 60),
    "green": (60, 180, 90),
    "blue": (60, 100, 220),
    "yellow": (230, 200, 60),
    "purple": (150, 70, 200),
    "orange": (230, 140, 50),
}
SHAPE_TYPES = ["circle", "square"]

FAR_DEPTH = 1.0
NEAR_DEPTH = 0.2
MIN_SHAPE_DEPTH = 0.05


def make_background(size):
    """Ground-plane style gradient: far (top) -> near (bottom)."""
    h, w = size
    depth = np.linspace(FAR_DEPTH, NEAR_DEPTH, h, dtype=np.float32)
    depth_map = np.tile(depth[:, None], (1, w))

    # Simple sky-to-ground gradient for the RGB background.
    top_color = np.array([180, 200, 225], dtype=np.float32)
    bottom_color = np.array([120, 150, 110], dtype=np.float32)
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    rgb = (1 - t) * top_color + t * bottom_color
    rgb = np.tile(rgb, (1, w, 1))
    return rgb.astype(np.uint8), depth_map


def draw_shapes(image, depth_map, num_shapes, rng):
    h, w = depth_map.shape
    draw = ImageDraw.Draw(image)
    prompt_parts = []

    for _ in range(num_shapes):
        shape_type = rng.choice(SHAPE_TYPES)
        color_name = rng.choice(list(SHAPE_COLORS.keys()))
        color = SHAPE_COLORS[color_name]

        radius = rng.randint(int(0.08 * w), int(0.18 * w))
        cx = rng.randint(radius, w - radius)
        cy = rng.randint(radius, h - radius)

        local_bg_depth = float(depth_map[cy, cx])
        offset = rng.uniform(0.1, 0.4)
        shape_depth = max(MIN_SHAPE_DEPTH, local_bg_depth - offset)

        if shape_type == "circle":
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(bbox, fill=color)
            mask = _circle_mask(h, w, cx, cy, radius)
        else:
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.rectangle(bbox, fill=color)
            mask = _square_mask(h, w, cx, cy, radius)

        depth_map[mask] = shape_depth
        article = "an" if color_name[0] in "aeiou" else "a"
        prompt_parts.append(f"{article} {color_name} {shape_type}")

    prompt = " and ".join(prompt_parts) if prompt_parts else "an empty scene"
    return image, depth_map, prompt


def _circle_mask(h, w, cx, cy, radius):
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _square_mask(h, w, cx, cy, radius):
    yy, xx = np.mgrid[0:h, 0:w]
    return (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)


def generate_split(out_dir, split, num_samples, image_size, seed):
    rng = random.Random(seed)
    images_dir = os.path.join(out_dir, "images")
    depth_dir = os.path.join(out_dir, "depth")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    manifest_path = os.path.join(out_dir, f"{split}.jsonl")
    with open(manifest_path, "w") as f:
        for i in range(num_samples):
            rgb, depth_map = make_background((image_size, image_size))
            image = Image.fromarray(rgb, mode="RGB")

            num_shapes = rng.randint(1, 3)
            image, depth_map, prompt = draw_shapes(image, depth_map, num_shapes, rng)

            image_name = f"{split}_{i:04d}.png"
            depth_name = f"{split}_{i:04d}.npy"
            image.save(os.path.join(images_dir, image_name))
            np.save(os.path.join(depth_dir, depth_name), depth_map.astype(np.float32))

            f.write(json.dumps({
                "image": image_name,
                "depth": depth_name,
                "prompt": prompt,  # bonus: usable later for text-guided depth experiments
            }) + "\n")

    print(f"{split}: wrote {num_samples} samples to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="data/synthetic_depth")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--train_samples", type=int, default=240)
    parser.add_argument("--val_samples", type=int, default=40)
    parser.add_argument("--test_samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_split(args.out_dir, "train", args.train_samples, args.image_size, args.seed)
    generate_split(args.out_dir, "val", args.val_samples, args.image_size, args.seed + 1)
    generate_split(args.out_dir, "test", args.test_samples, args.image_size, args.seed + 2)


if __name__ == "__main__":
    main()
