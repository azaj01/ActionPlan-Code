"""
Render SMPL mesh video from a .npy file.

Supports three input formats (detected by feature dimension):
- 16: motion latents only (TAE-decoded to 272, then SMPL)
- 32: motion latents + text latents (first 16 dims used, same as above)
- 272: decoded motion features (directly converted to SMPL)

Usage:
    python render.py path/to/motion.npy [--out_path OUTPUT] [--fps 30] [--tae_checkpoint PATH]
"""

import argparse
import logging
import os
import sys

import numpy as np

# Ensure ActionPlan-Code root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


def T(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))


def _npy_to_vertices(
    data: np.ndarray,
    tae_checkpoint: str | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Convert .npy data to SMPL vertices based on feature dimension.

    Args:
        data: Loaded numpy array, shape (T, D) or (1, T, D) where D in {16, 32, 272}
        tae_checkpoint: Path to TAE checkpoint (for 16/32 dim). Uses default if None.
        device: Device for TAE and SMPL.

    Returns:
        vertices: (T, 6890, 3) SMPL vertices
    """
    import torch

    from src.tae import decode_latents, load_tae
    from src.tools.streamer272_feats import streamer272_to_smpl
    from src.tools.smpl_layer import SMPLH

    # Squeeze batch dim if present
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]

    nfeats = data.shape[-1]

    tae_model = None
    if nfeats in (16, 32) and tae_checkpoint:
        tae_model = load_tae(checkpoint_path=tae_checkpoint, device=device)

    if nfeats == 16:
        # Motion latents only: decode via TAE -> 272 -> SMPL
        latents = np.asarray(data, dtype=np.float32)
        decoded = decode_latents(
            latents,
            model=tae_model,
            device=device,
            remove_reference_token=False,
            denormalize=True,
        )
        if isinstance(decoded, torch.Tensor):
            decoded = decoded.cpu().numpy()
        motion_272 = decoded

    elif nfeats == 32:
        # Motion + text latents: use first 16 dims (motion only)
        latents = np.asarray(data[..., :16], dtype=np.float32)
        decoded = decode_latents(
            latents,
            model=tae_model,
            device=device,
            remove_reference_token=False,
            denormalize=True,
        )
        if isinstance(decoded, torch.Tensor):
            decoded = decoded.cpu().numpy()
        motion_272 = decoded

    elif nfeats == 272:
        # Already decoded motion features
        motion_272 = np.asarray(data, dtype=np.float32)

    else:
        raise ValueError(
            f"Expected feature dimension 16, 32, or 272, got {nfeats}. "
            "Supported formats: 16 (motion latents), 32 (motion+text latents), 272 (decoded motion)."
        )

    # 272 -> SMPL vertices
    dev = torch.device(device)
    motion_tensor = torch.from_numpy(motion_272).float().to(dev)
    smpl_data = streamer272_to_smpl(motion_tensor)
    poses = smpl_data["poses"].to(dev)
    trans = smpl_data["trans"].to(dev)

    actionplan_root = os.path.dirname(os.path.abspath(__file__))
    smplh_path = os.path.join(actionplan_root, "deps", "smplh")
    smpl_layer = SMPLH(
        path=smplh_path,
        jointstype="vertices",
        input_pose_rep="axisangle",
        gender="neutral",
        batch_size=512,
    ).to(dev)

    vertices = smpl_layer(poses, trans).cpu().numpy()
    return vertices


def main():
    parser = argparse.ArgumentParser(
        description="Render SMPL mesh video from .npy (16/32/272 dim)"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to .npy file (16=motion latents, 32=motion+text latents, 272=decoded motion)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output video path (default: input path with .mp4 extension)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        help="Output extension (default: mp4)",
    )
    parser.add_argument(
        "--tae_checkpoint",
        type=str,
        default=None,
        help="TAE checkpoint path for 16/32 dim decoding (uses default if not set)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/mps/cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--y_is_z_axis",
        action="store_true",
        help="Apply Y-up to Z-up coordinate transform",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    path = args.path
    if not os.path.isfile(path):
        logger.error("File not found: %s", path)
        sys.exit(1)

    motions = np.load(path)

    # Squeeze single-sample batch
    if motions.ndim == 3 and motions.shape[0] == 1:
        motions = motions[0]

    nfeats = motions.shape[-1]

    if nfeats in (16, 32, 272):
        # Latent or 272-dim: decode to vertices
        device = args.device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        vertices = _npy_to_vertices(
            motions,
            tae_checkpoint=args.tae_checkpoint,
            device=device,
        )
        motions = vertices
        use_smpl_renderer = True
        convert_yup_to_zup = True
    elif motions.ndim == 3 and motions.shape[1] == 6890:
        # Already SMPL vertices (T, 6890, 3)
        use_smpl_renderer = True
        convert_yup_to_zup = False
    else:
        # Joints or other format: use joints renderer
        use_smpl_renderer = False
        convert_yup_to_zup = False

    ext = "." + args.ext.replace(".", "")
    out_path = args.out_path
    if out_path is None:
        out_path = os.path.splitext(path)[0] + ext

    logger.info("Rendering video to: %s", out_path)

    if motions.ndim == 3 and motions.shape[0] == 1:
        motions = motions[0]

    if args.y_is_z_axis:
        x, mz, my = T(motions)
        motions = T(np.stack((x, -my, mz), axis=0))

    if use_smpl_renderer:
        from src.renderer.humor import HumorRenderer
        renderer = HumorRenderer(fps=args.fps, convert_yup_to_zup=convert_yup_to_zup)
    else:
        from src.renderer.matplotlib import MatplotlibRender
        renderer = MatplotlibRender()

    renderer(motions, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
