# Reference (upstream):
#   Repository: https://github.com/Chrixtar/SRM
#   Source: blob/main/src/model/flow/rectified_flow.py
#   Notes: Small helpers for rectified-flow / masking; project-local (no direct upstream twin).

"""Shared utilities for rectified flow models."""

import torch


def masked(tensor, mask):
    """Apply a boolean mask to zero-out padded regions, with safe broadcasting.

    Supports tensors of shape [B, T, D] with masks [B, T], and tensors of shape
    [B, D] with masks [B, 1] or [B]. If a list of tensors is provided, applies
    the same masking to each element.
    """
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]

    # Ensure boolean mask
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)

    # Expand mask along trailing dims to enable broadcasting
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)

    return tensor * mask
