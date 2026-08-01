import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Standard ImageNet stats. For production use, prefer
# transformers.AutoImageProcessor.from_pretrained(model_name) to match a
# specific checkpoint's exact preprocessing -- we do it by hand here so the
# preprocessing steps are visible rather than hidden behind a processor.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DepthDataset(Dataset):
    def __init__(self, data_dir, split, image_size=224):
        self.data_dir = data_dir
        manifest_path = os.path.join(data_dir, f"{split}.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"{manifest_path} not found. Run "
                "`python data/generate_synthetic_depth.py` first."
            )
        with open(manifest_path) as f:
            self.entries = [json.loads(line) for line in f if line.strip()]

        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        image_path = os.path.join(self.data_dir, "images", entry["image"])
        depth_path = os.path.join(self.data_dir, "depth", entry["depth"])

        image = Image.open(image_path).convert("RGB")
        depth = np.load(depth_path).astype(np.float32)

        pixel_values = self.transform(image)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)

        return pixel_values, depth_tensor
