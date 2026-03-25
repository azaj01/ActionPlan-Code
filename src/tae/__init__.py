"""
Temporal AutoEncoder (TAE) from MotionStreamer.

This module provides the Causal TAE for encoding/decoding motion sequences
between 272-dim motion representation and 16-dim latent space.

The TAE applies 4x temporal compression:
- Input: 272-dim motion at 30 FPS
- Output: 16-dim latent at 7.5 FPS

IMPORTANT: The TAE was trained with z-normalized 272-dim features.
- Use normalize=True when encoding raw 272-dim motion
- Use denormalize=True when decoding to get actual 272-dim values

Reference: https://github.com/Li-xingXiao/272-dim-Motion-Representation
"""

from .tae import Causal_TAE, Causal_HumanTAE
from .loader import load_tae, decode_latents, encode_motion, load_norm_stats

__all__ = [
    "Causal_TAE",
    "Causal_HumanTAE", 
    "load_tae",
    "decode_latents",
    "encode_motion",
    "load_norm_stats",
]

