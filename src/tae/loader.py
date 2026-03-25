"""
TAE loader utilities.
"""

import os
import torch
import numpy as np
from typing import Union, Optional
from pathlib import Path

from .tae import Causal_HumanTAE


# Default checkpoint path relative to ActionPlan-Code root
DEFAULT_CHECKPOINT = "models/Causal_TAE/net_last.pth"

# Default normalization stats path (mean/std for 272-dim features)
DEFAULT_NORM_STATS_DIR = "datasets/motions/humanml3d_272/mean_std"

# Default TAE hyperparameters (matching the trained model)
DEFAULT_CONFIG = {
    "hidden_size": 1024,
    "down_t": 2,
    "stride_t": 2,
    "depth": 3,
    "dilation_growth_rate": 3,
    "activation": "relu",
    "latent_dim": 16,
    "clip_range": [-30, 20],
}


def load_norm_stats(stats_dir: Optional[str] = None) -> tuple:
    """Load normalization statistics (mean, std) for 272-dim features.
    
    Args:
        stats_dir: Directory containing Mean.npy and Std.npy files.
                   If None, uses default path.
    
    Returns:
        Tuple of (mean, std) as numpy arrays of shape (272,)
    """
    if stats_dir is None:
        src_dir = Path(__file__).parent.parent.parent  # ActionPlan-Code root
        stats_dir = src_dir / DEFAULT_NORM_STATS_DIR
    
    stats_dir = Path(stats_dir)
    mean_path = stats_dir / "Mean.npy"
    std_path = stats_dir / "Std.npy"
    
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found at {stats_dir}. "
            f"Expected Mean.npy and Std.npy files."
        )
    
    mean = np.load(mean_path)
    std = np.load(std_path)
    
    return mean, std


def load_tae(
    checkpoint_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    **config_overrides
) -> Causal_HumanTAE:
    """Load a pretrained TAE model.
    
    Args:
        checkpoint_path: Path to the checkpoint file. If None, uses default path.
        device: Device to load the model on.
        **config_overrides: Override default config parameters.
    
    Returns:
        Loaded Causal_HumanTAE model in eval mode.
    """
    # Resolve checkpoint path
    if checkpoint_path is None:
        # Try to find the default checkpoint relative to this file
        src_dir = Path(__file__).parent.parent.parent  # ActionPlan-Code root
        checkpoint_path = src_dir / DEFAULT_CHECKPOINT
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TAE checkpoint not found at {checkpoint_path}")
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update(config_overrides)
    
    # Create model
    model = Causal_HumanTAE(
        hidden_size=config["hidden_size"],
        down_t=config["down_t"],
        stride_t=config["stride_t"],
        depth=config["depth"],
        dilation_growth_rate=config["dilation_growth_rate"],
        activation=config["activation"],
        latent_dim=config["latent_dim"],
        clip_range=config["clip_range"],
    )
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "net" in ckpt:
        model.load_state_dict(ckpt["net"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    
    # Move to device and set eval mode
    model = model.to(device)
    model.eval()
    
    print(f"Loaded TAE from {checkpoint_path}")
    return model


def decode_latents(
    latents: Union[np.ndarray, torch.Tensor],
    model: Optional[Causal_HumanTAE] = None,
    device: Union[str, torch.device] = "cpu",
    remove_reference_token: bool = True,
    denormalize: bool = True,
    norm_stats_dir: Optional[str] = None,
) -> torch.Tensor:
    """Decode latent representations to 272-dim motion.
    
    The TAE was trained with normalized 272-dim features (z-normalization).
    The decoder outputs normalized values, so we need to denormalize to get
    the actual motion features.
    
    Args:
        latents: Latent tensor of shape (seq_len, 16) or (batch, seq_len, 16)
        model: TAE model. If None, loads default checkpoint.
        device: Device to use for decoding.
        remove_reference_token: If True, removes the last token (reference_end_latent)
            before decoding.
        denormalize: If True, applies inverse normalization to get actual 272-dim values.
                     Set to False if you want the raw normalized output.
        norm_stats_dir: Directory containing Mean.npy and Std.npy for denormalization.
                        If None, uses default path.
    
    Returns:
        Decoded motion of shape (batch, seq_len*4, 272) or (seq_len*4, 272)
    """
    # Load model if not provided
    if model is None:
        model = load_tae(device=device)
    
    # Convert to tensor if needed
    if isinstance(latents, np.ndarray):
        latents = torch.from_numpy(latents).float()
    
    # Track if we need to squeeze batch dim later
    squeeze_batch = False
    
    # Ensure batch dimension
    if latents.ndim == 2:
        latents = latents.unsqueeze(0)
        squeeze_batch = True
    
    # Remove reference_end_latent if requested
    if remove_reference_token:
        latents = latents[:, :-1, :]
    
    # Move to device
    latents = latents.to(device)
    
    # Decode
    with torch.no_grad():
        decoded = model.forward_decoder(latents)
    
    # Move back to CPU
    decoded = decoded.cpu()
    
    # Denormalize to get actual 272-dim values
    if denormalize:
        mean, std = load_norm_stats(norm_stats_dir)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()
        decoded = decoded * std + mean
    
    # Remove batch dim if we added it
    if squeeze_batch:
        decoded = decoded.squeeze(0)
    
    return decoded


def encode_motion(
    motion: Union[np.ndarray, torch.Tensor],
    model: Optional[Causal_HumanTAE] = None,
    device: Union[str, torch.device] = "cpu",
    normalize: bool = True,
    norm_stats_dir: Optional[str] = None,
) -> torch.Tensor:
    """Encode 272-dim motion to latent representation.
    
    The TAE expects normalized 272-dim features (z-normalization).
    By default, this function normalizes the input before encoding.
    
    Args:
        motion: Motion tensor of shape (seq_len, 272) or (batch, seq_len, 272)
        model: TAE model. If None, loads default checkpoint.
        device: Device to use for encoding.
        normalize: If True, normalizes the input before encoding.
                   Set to False if input is already normalized.
        norm_stats_dir: Directory containing Mean.npy and Std.npy for normalization.
                        If None, uses default path.
    
    Returns:
        Latent representation of shape (batch, seq_len//4, 16) or (seq_len//4, 16)
    """
    # Load model if not provided
    if model is None:
        model = load_tae(device=device)
    
    # Convert to tensor if needed
    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion).float()
    
    # Track if we need to squeeze batch dim later
    squeeze_batch = False
    
    # Ensure batch dimension
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
        squeeze_batch = True
    
    # Normalize input
    if normalize:
        mean, std = load_norm_stats(norm_stats_dir)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()
        motion = (motion - mean) / std
    
    # Move to device
    motion = motion.to(device)
    
    # Encode
    with torch.no_grad():
        latent, _, _ = model.encode(motion)
    
    # Reshape latent from (batch*seq, 16) to (batch, seq, 16)
    batch_size = motion.shape[0]
    seq_len = latent.shape[0] // batch_size
    latent = latent.view(batch_size, seq_len, -1)
    
    # Move back to CPU
    latent = latent.cpu()
    
    # Remove batch dim if we added it
    if squeeze_batch:
        latent = latent.squeeze(0)
    
    return latent

