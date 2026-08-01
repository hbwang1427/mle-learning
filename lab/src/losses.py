"""Depth regression losses.

l1_loss is the simplest thing that works and is what the training loop uses
by default. scale_invariant_loss is the Eigen et al. (2014) loss mentioned
in Class 1 -- it penalizes shape errors more than a global scale/offset
mismatch, which matters more for relative (non-metric) depth estimation.
"""
import torch


def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def scale_invariant_loss(pred, target, eps=1e-6, lambda_weight=0.5):
    """Eigen et al. scale-invariant log loss: mean(d^2) - lambda * mean(d)^2."""
    log_pred = torch.log(pred.clamp(min=eps))
    log_target = torch.log(target.clamp(min=eps))
    diff = log_pred - log_target

    term1 = torch.mean(diff ** 2)
    per_sample_mean = diff.flatten(1).mean(dim=1)
    term2 = torch.mean(per_sample_mean ** 2)
    return term1 - lambda_weight * term2


LOSS_REGISTRY = {
    "l1": l1_loss,
    "silog": scale_invariant_loss,
}


def get_loss_fn(name):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_fn '{name}'. Options: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name]
