"""Frozen ViT encoder (Hugging Face transformers) + a small trainable conv
head for dense depth regression.

This mirrors the "frozen backbone + lightweight trainable head" pattern used
throughout this course (see ml-multimodal's FusionCLIPModel): the pretrained
ViT never updates, so training is cheap and the head is the only thing you
version, checkpoint, and ship. Architecturally this is a minimal version of
DPT (Dense Prediction Transformer) from Class 1: ViT encoder -> reassemble
patch tokens into a feature map -> a small decoder head -> upsample to a
depth map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ViTDepthModel(nn.Module):
    def __init__(self, vision_model_name, freeze_backbone=True, head_hidden_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(vision_model_name)
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(self.hidden_size, head_hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_hidden_dim, head_hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_hidden_dim // 2, 1, kernel_size=1),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()  # keep frozen backbone's BatchNorm/Dropout inactive
        return self

    def forward(self, pixel_values):
        """pixel_values: (B, 3, H, W) -> depth: (B, H, W) in [0, 1]."""
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            outputs = self.encoder(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state  # (B, 1 + N_patches, hidden_size)
        patch_tokens = tokens[:, 1:, :]  # drop the CLS token

        b, n, c = patch_tokens.shape
        h_img, w_img = pixel_values.shape[-2:]
        h_p, w_p = h_img // self.patch_size, w_img // self.patch_size
        assert h_p * w_p == n, (
            f"Patch grid {h_p}x{w_p}={h_p * w_p} doesn't match {n} patch tokens "
            f"-- check that image_size is divisible by patch_size ({self.patch_size})."
        )

        feature_map = patch_tokens.permute(0, 2, 1).reshape(b, c, h_p, w_p)
        depth_logits = self.head(feature_map)  # (B, 1, h_p, w_p)
        depth = torch.sigmoid(depth_logits)  # normalize to [0, 1], matches synthetic GT range

        depth_full = F.interpolate(
            depth, size=(h_img, w_img), mode="bilinear", align_corners=False
        )
        return depth_full.squeeze(1)  # (B, H, W)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total
