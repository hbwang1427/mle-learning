"""Depth evaluation metrics, matching the ones named in the Class 1 notes
(Course Project Preview): AbsRel, RMSE, and delta-threshold accuracy.
"""
import torch


def abs_rel(pred, target, eps=1e-6):
    return torch.mean(torch.abs(pred - target) / target.clamp(min=eps)).item()


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def delta_accuracy(pred, target, threshold=1.25, eps=1e-6):
    ratio = torch.maximum(pred / target.clamp(min=eps), target / pred.clamp(min=eps))
    return torch.mean((ratio < threshold).float()).item()


def compute_all(pred, target):
    return {
        "abs_rel": abs_rel(pred, target),
        "rmse": rmse(pred, target),
        "delta1_25": delta_accuracy(pred, target, 1.25),
    }
