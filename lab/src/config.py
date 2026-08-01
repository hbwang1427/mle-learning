from dataclasses import dataclass


@dataclass
class Config:
    # Vision backbone (any HF AutoModel exposing a ViT-style last_hidden_state).
    # Default is a small ViT so the lab trains fast on a laptop CPU.
    # Swap to "google/vit-base-patch16-224-in21k" for a full-size ViT.
    vision_model_name: str = "WinKawaks/vit-tiny-patch16-224"
    freeze_backbone: bool = True

    image_size: int = 224
    head_hidden_dim: int = 128

    data_dir: str = "data/synthetic_depth"
    output_dir: str = "outputs"

    loss_fn: str = "l1"  # "l1" or "silog" (see src/losses.py)
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 5
    num_workers: int = 2
    seed: int = 42
