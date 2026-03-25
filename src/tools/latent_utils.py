"""
Shared helpers for sampling with the latent diffusion model + TAE decoder.

Usage pattern:
- Instantiate `LatentPipeline` once (reuses SMPL + renderer).
- Run a sampler with `render=False` to get latent features.
- Call `pipeline.decode_and_render(...)` to decode to 272-d motion and render.

Defaults:
- Latent diffusion checkpoint: the fixed absolute path provided by the user.
- TAE checkpoint: falls back to the built-in default (models/Causal_TAE/net_last.pth).
- Latent FPS: 7.5 (latent space), decoded/rendered FPS: 30.0.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from src.tae import decode_latents, load_tae
from src.tools.streamer272_feats import streamer272_to_smpl
from src.tools.smpl_layer import SMPLH
from src.renderer.humor import HumorRenderer

# Latent sampling and decoded render FPS
LATENT_FPS = 7.5  # latent model timestep rate
DECODED_FPS = 30.0  # decoder output / render rate


def derive_run_dir_from_ckpt(ckpt_path: str) -> Path:
    """Return run_dir assuming standard logs/checkpoints/<ckpt> layout."""
    ckpt = Path(ckpt_path).expanduser().resolve()
    return ckpt.parent.parent.parent


def _ensure_headless_gl():
    if os.environ.get("DISPLAY") in (None, ""):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")


class LatentPipeline:
    """Decode latent samples to 272-d motion and render via SMPL."""

    def __init__(self, tae_checkpoint: Optional[str], device: str = "cuda") -> None:
        _ensure_headless_gl()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # Lazy load TAE
        self.tae_model = load_tae(checkpoint_path=tae_checkpoint, device=self.device)

        # Build SMPL + renderer once
        actionplan_root = Path(__file__).resolve().parent.parent.parent
        smplh_path = actionplan_root / "deps" / "smplh"
        self.smpl_layer = SMPLH(
            path=str(smplh_path),
            jointstype="vertices",
            input_pose_rep="axisangle",
            gender="neutral",
            batch_size=512,
        )
        # Latent model outputs Y-up, so we need to convert to Z-up for visualization
        self.renderer = HumorRenderer(fps=DECODED_FPS, convert_yup_to_zup=True)

    def decode_latents(self, latents: np.ndarray | torch.Tensor) -> np.ndarray:
        """Decode latent sequence -> 272-d motion (numpy)."""
        decoded = decode_latents(
            latents,
            model=self.tae_model,
            device=self.device,
            remove_reference_token=False,
            denormalize=True,
        )
        if isinstance(decoded, torch.Tensor):
            decoded = decoded.cpu().numpy()
        return decoded

    def render_motion(
        self,
        motion_272: np.ndarray,
        output_path: str,
        text_overlay: str = "",
        text_overlay_seq: Optional[Union[List[str], np.ndarray]] = None,
        vertex_color: Optional[list] = None,
        frame_vertex_colors: Optional[np.ndarray] = None,
    ) -> str:
        """Render decoded 272-d motion to MP4.
        text_overlay_seq: per-frame text (one string per decoded frame); overrides text_overlay when set.
        """
        motion_tensor = torch.from_numpy(motion_272).float()
        smpl_data = streamer272_to_smpl(motion_tensor)
        poses = smpl_data["poses"]
        trans = smpl_data["trans"]
        vertices = self.smpl_layer(poses, trans).cpu().numpy()

        render_kwargs: Dict[str, object] = {
            "vertices": vertices,
            "output": output_path,
            "text_overlay": text_overlay,
        }
        if text_overlay_seq is not None:
            render_kwargs["text_overlay_seq"] = list(text_overlay_seq)
        if vertex_color is not None:
            render_kwargs["vertex_color"] = vertex_color
        if frame_vertex_colors is not None:
            render_kwargs["frame_vertex_colors"] = frame_vertex_colors

        self.renderer(**render_kwargs)
        return output_path

    def decode_and_render(
        self,
        latents: np.ndarray | torch.Tensor,
        output_root: Path,
        base_name: str,
        prompt: str,
        render: bool = True,
    ) -> Dict[str, str]:
        """Decode latents, save artifacts, optionally render video with prompt overlay. Returns paths."""
        output_root.mkdir(parents=True, exist_ok=True)
        latents_np = latents if isinstance(latents, np.ndarray) else latents.detach().cpu().numpy()

        latents_path = output_root / f"{base_name}_latents.npy"
        np.save(latents_path, latents_np)

        decoded_272 = self.decode_latents(latents_np)
        decoded_path = output_root / f"{base_name}_decoded272.npy"
        np.save(decoded_path, decoded_272)

        video_path = output_root / f"{base_name}_latent.mp4"
        if render:
            self.render_motion(
                decoded_272,
                output_path=str(video_path),
                text_overlay=prompt,
            )

        metadata_path = output_root / f"{base_name}_latent_meta.txt"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(f"prompt: {prompt}\n")
            f.write(f"latent_fps: {LATENT_FPS}\n")
            f.write(f"decoded_fps: {DECODED_FPS}\n")
            f.write(f"latents_shape: {latents_np.shape}\n")
            f.write(f"decoded_shape: {decoded_272.shape}\n")

        return {
            "video_path": str(video_path),
            "latents_path": str(latents_path),
            "decoded_path": str(decoded_path),
            "metadata_path": str(metadata_path),
        }

__all__ = [
    "LatentPipeline",
    "LATENT_FPS",
    "DECODED_FPS",
    "derive_run_dir_from_ckpt",
]
