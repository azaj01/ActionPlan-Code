"""
Sample Logger for ActionPlan 32-dim representation (16 motion + 16 CLIP).

This callback generates and saves motion samples during training,
specifically designed for the ActionPlan 32-dim latent representation:
- First 16 features: motion latents
- Last 16 features: compressed CLIP embeddings (AE-16)

The callback displays both generated and ground truth samples, along with
the retrieved text labels corresponding to the CLIP embeddings.
Videos show label boxes at the bottom that highlight active labels per frame.

Supports two CLIP decoding modes:
- "ae": CLIPAutoencoder decode + Cosine KNN
"""

import os
import sys
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import torch
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from sklearn.neighbors import NearestNeighbors

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


from src.renderer.video import get_drive_safe_videofile_kwargs

logger = logging.getLogger(__name__)


# Default paths relative to project root
ACTIONPLAN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_AE_PATH = os.path.join(
    ACTIONPLAN_ROOT, "datasets", "embeddings", "temp_clip", "clip_enc_latent_aligned_ae16", "autoencoder.pt"
)
DEFAULT_AE_FALLBACK_PATH = os.path.join(
    ACTIONPLAN_ROOT, "outputs", "clip_autoencoder", "best.pt"
)
DEFAULT_CLIP_EMBEDDINGS_PATH = os.path.join(
    ACTIONPLAN_ROOT, "datasets", "embeddings", "raw", "clip_embeddings.tsv"
)
DEFAULT_LABEL_ID_PATH = os.path.join(
    ACTIONPLAN_ROOT, "datasets", "embeddings", "raw", "label_to_id.json"
)


def _move_tx_to_device(tx: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if tx is None:
        return None
    out = {}
    for k, v in tx.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _slice_first_dim_if_batched(d: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Return a shallow-copied dict where any batched tensors are sliced on first dim."""
    if d is None:
        return None
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] >= end:
            out[k] = v[start:end]
        else:
            out[k] = v
    return out


def create_side_by_side_video(video1_path, video2_path, output_path, fps=30.0):
    """Create a side-by-side comparison video using moviepy."""
    try:
        import moviepy.editor as mp

        video1 = mp.VideoFileClip(video1_path, audio=False)
        video2 = mp.VideoFileClip(video2_path, audio=False)

        min_duration = min(video1.duration, video2.duration)
        video1 = video1.subclip(0, min_duration)
        video2 = video2.subclip(0, min_duration)

        side_by_side = mp.clips_array([[video1, video2]])

        side_by_side.write_videofile(
            output_path,
            **get_drive_safe_videofile_kwargs(fps=fps),
        )

        video1.close()
        video2.close()
        side_by_side.close()

        return True
    except Exception as e:
        logger.warning(f"SampleLoggerActionPlan: failed to create side-by-side video: {e}")
        return False


class CLIPTextRetriever:
    """Retrieves text labels from compressed CLIP embeddings using KNN.

    Supports AE mode:
      - ``"ae"``: loads a trained ``CLIPAutoencoder``, calls ``decode``
        to recover 512-d, then cosine KNN (matching AE training).
    """

    def __init__(
        self,
        mode: str = "ae",
        autoencoder_path: str = DEFAULT_AE_PATH,
        clip_embeddings_path: str = DEFAULT_CLIP_EMBEDDINGS_PATH,
        label_id_path: str = DEFAULT_LABEL_ID_PATH,
    ):
        mode = str(mode).lower()
        if mode != "ae":
            logger.warning("CLIPTextRetriever: mode '%s' is deprecated; falling back to 'ae'.", mode)
            mode = "ae"
        self.mode = mode
        self.autoencoder_path = autoencoder_path
        self.clip_embeddings_path = clip_embeddings_path
        self.label_id_path = label_id_path

        self._ae_model = None
        self._ae_device = None
        self._knn = None
        self._inverse_texts_dict = None
        self._loaded = False

    @staticmethod
    def _resolve_existing_path(path_value: Optional[str], fallback_paths: Optional[List[str]] = None) -> Optional[str]:
        """Resolve a path against common roots and return an existing absolute path."""
        candidates: List[str] = []
        if path_value:
            raw = os.path.expanduser(str(path_value))
            if os.path.isabs(raw):
                candidates.append(raw)
            else:
                candidates.extend(
                    [
                        os.path.join(ACTIONPLAN_ROOT, raw),
                        os.path.join(os.getcwd(), raw),
                    ]
                )
        if fallback_paths:
            candidates.extend([str(p) for p in fallback_paths])

        seen = set()
        for cand in candidates:
            abs_cand = os.path.abspath(cand)
            if abs_cand in seen:
                continue
            seen.add(abs_cand)
            if os.path.exists(abs_cand):
                return abs_cand
        return None

    # ------------------------------------------------------------------ #
    # lazy loading
    # ------------------------------------------------------------------ #
    def _load(self):
        """Lazy load the decoder (PCA or AE), KNN index, and label mapping."""
        if self._loaded:
            return True

        try:
            # ── load decoder ──────────────────────────────────────────
            resolved_ae = self._resolve_existing_path(
                self.autoencoder_path,
                fallback_paths=[DEFAULT_AE_PATH, DEFAULT_AE_FALLBACK_PATH],
            )
            if resolved_ae is None:
                logger.warning(f"CLIPTextRetriever: AE checkpoint not found at {self.autoencoder_path}")
                return False
            self.autoencoder_path = resolved_ae
            self._ae_model, self._ae_device = self._load_autoencoder(self.autoencoder_path)
            logger.info(f"CLIPTextRetriever[ae]: loaded autoencoder from {self.autoencoder_path}")

            # ── load CLIP label bank ──────────────────────────────────
            resolved_clip_embeddings = self._resolve_existing_path(
                self.clip_embeddings_path,
                fallback_paths=[DEFAULT_CLIP_EMBEDDINGS_PATH],
            )
            if resolved_clip_embeddings is None:
                logger.warning(f"CLIPTextRetriever: CLIP embeddings not found at {self.clip_embeddings_path}")
                return False
            self.clip_embeddings_path = resolved_clip_embeddings

            logger.info(f"CLIPTextRetriever: loading CLIP embeddings from {self.clip_embeddings_path}")
            embeddings_np = np.loadtxt(self.clip_embeddings_path, delimiter='\t')
            logger.info(f"CLIPTextRetriever: loaded embeddings with shape {embeddings_np.shape}")

            # KNN metric depends on mode
            metric = "cosine"
            self._knn = NearestNeighbors(n_neighbors=1, metric=metric)
            self._knn.fit(embeddings_np)
            logger.info(f"CLIPTextRetriever: fitted KNN ({metric}) on CLIP embeddings")

            # ── load label mapping ────────────────────────────────────
            resolved_label_path = self._resolve_existing_path(
                self.label_id_path,
                fallback_paths=[DEFAULT_LABEL_ID_PATH],
            )
            if resolved_label_path is None:
                logger.warning(f"CLIPTextRetriever: label mapping not found at {self.label_id_path}")
                return False
            self.label_id_path = resolved_label_path

            import json
            with open(self.label_id_path, 'r') as f:
                texts_dict = json.load(f)
            self._inverse_texts_dict = {v: k for k, v in texts_dict.items()}
            logger.info(f"CLIPTextRetriever: loaded {len(self._inverse_texts_dict)} text labels")

            self._loaded = True
            return True

        except Exception as e:
            logger.warning(f"CLIPTextRetriever: failed to load resources: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def _load_autoencoder(ckpt_path: str):
        """Load a CLIPAutoencoder from a checkpoint dict."""
        # Add the clip_autoencoder package to sys.path so we can import the model
        ae_pkg_dir = os.path.join(ACTIONPLAN_ROOT, "models", "clip_autoencoder")
        if ae_pkg_dir not in sys.path:
            sys.path.insert(0, ae_pkg_dir)
        from model import CLIPAutoencoder  # noqa: E402

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        ae = CLIPAutoencoder(
            input_dim=cfg.get("input_dim", 512),
            latent_dim=cfg.get("latent_dim", 16),
            hidden_dims=cfg.get("hidden_dims", [256, 128]),
            dropout=cfg.get("dropout", 0.0),
        )
        ae.load_state_dict(ckpt["model_state_dict"])
        device = torch.device("cpu")
        ae = ae.to(device).eval()
        return ae, device

    # ------------------------------------------------------------------ #
    # 16-d  →  512-d
    # ------------------------------------------------------------------ #
    def _decode_to_512(self, clip_16: np.ndarray) -> np.ndarray:
        """Decode 16-d compressed CLIP embeddings back to 512-d.

        Args:
            clip_16: ``[T, 16]``

        Returns:
            ``[T, 512]`` reconstructed embeddings
        """
        with torch.no_grad():
            x = torch.from_numpy(clip_16).float().to(self._ae_device)
            recon = self._ae_model.decode(x)
        return recon.cpu().numpy()

    # ------------------------------------------------------------------ #
    # public API  (unchanged signatures)
    # ------------------------------------------------------------------ #
    def retrieve_texts_all_frames(
        self,
        clip_embeds_16: np.ndarray,
    ) -> Tuple[List[str], List[int]]:
        """
        Retrieve text labels for ALL frames from 16-d CLIP embeddings.

        Args:
            clip_embeds_16: [T, 16] compressed CLIP embeddings

        Returns:
            texts: List of retrieved text labels (one per frame)
            indices: List of corresponding neighbor indices
        """
        if not self._load():
            return [], []

        try:
            clip_embeds_full = self._decode_to_512(clip_embeds_16)  # [T, 512]
            distances, indices = self._knn.kneighbors(clip_embeds_full)

            texts = []
            neighbor_indices = []
            for frame_neighbors in indices:
                neighbor_idx = frame_neighbors[0]
                text = self._inverse_texts_dict.get(neighbor_idx, f"<unknown:{neighbor_idx}>")
                texts.append(text)
                neighbor_indices.append(neighbor_idx)

            return texts, neighbor_indices

        except Exception as e:
            logger.warning(f"CLIPTextRetriever: failed to retrieve texts: {e}")
            traceback.print_exc()
            return [], []

    def retrieve_texts(
        self,
        clip_embeds_16: np.ndarray,
        sample_indices: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Retrieve text labels from 16-d CLIP embeddings.

        Args:
            clip_embeds_16: [T, 16] compressed CLIP embeddings
            sample_indices: Optional list of frame indices to sample

        Returns:
            texts: List of retrieved text labels
            indices: List of corresponding neighbor indices
        """
        if not self._load():
            n = len(sample_indices) if sample_indices is not None else (
                int(clip_embeds_16.shape[0]) if hasattr(clip_embeds_16, "shape") and len(clip_embeds_16.shape) > 0 else 0
            )
            return ["<clip-decoder-unavailable>"] * n, [-1] * n

        try:
            clip_embeds_full = self._decode_to_512(clip_embeds_16)  # [T, 512]

            if sample_indices is not None:
                clip_embeds_full = clip_embeds_full[sample_indices]

            distances, indices = self._knn.kneighbors(clip_embeds_full)

            texts = []
            neighbor_indices = []
            for frame_neighbors in indices:
                for neighbor_idx in frame_neighbors:
                    text = self._inverse_texts_dict.get(neighbor_idx, f"<unknown:{neighbor_idx}>")
                    texts.append(text)
                    neighbor_indices.append(neighbor_idx)

            return texts, neighbor_indices

        except Exception as e:
            logger.warning(f"CLIPTextRetriever: failed to retrieve texts: {e}")
            traceback.print_exc()
            return [], []

    def get_summary_text(
        self,
        clip_embeds_16: np.ndarray,
        num_samples: int = 5,
    ) -> str:
        """Get a summary of retrieved texts by sampling frames."""
        T = clip_embeds_16.shape[0]
        if T == 0:
            return ""

        sample_indices = np.linspace(0, T - 1, min(num_samples, T), dtype=int).tolist()
        texts, _ = self.retrieve_texts(clip_embeds_16, sample_indices)

        if not texts:
            return ""

        seen = set()
        unique_texts = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                unique_texts.append(t)

        return " → ".join(unique_texts[:5])


def add_label_boxes_to_frame(
    frame: np.ndarray,
    all_labels: List[str],
    current_label: str,
    box_height: int = 60,
    max_labels_per_row: int = 6,
    font_size: int = 14,
) -> np.ndarray:
    """
    Add label boxes at the bottom of a frame.
    
    Args:
        frame: [H, W, 3] RGB image as numpy array
        all_labels: List of all unique labels to display
        current_label: The label that is currently active (will be highlighted)
        box_height: Height of the label box area
        max_labels_per_row: Maximum labels per row
        font_size: Font size for labels
    
    Returns:
        New frame with label boxes added at the bottom
    """
    if not HAS_PIL:
        return frame
    
    H, W, C = frame.shape
    
    # Create PIL image
    img = Image.fromarray(frame)
    
    # Calculate rows needed
    n_labels = len(all_labels)
    n_rows = (n_labels + max_labels_per_row - 1) // max_labels_per_row
    total_box_height = box_height * n_rows
    
    # Create new image with extra space at bottom
    new_img = Image.new('RGB', (W, H + total_box_height), color=(40, 40, 40))
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()
    
    # Calculate box dimensions
    padding = 8
    box_margin = 4
    
    for i, label in enumerate(all_labels):
        row = i // max_labels_per_row
        col = i % max_labels_per_row
        
        # Calculate box width based on available space
        labels_in_row = min(max_labels_per_row, n_labels - row * max_labels_per_row)
        box_width = (W - (labels_in_row + 1) * box_margin) // labels_in_row
        
        x0 = box_margin + col * (box_width + box_margin)
        y0 = H + row * box_height + box_margin
        x1 = x0 + box_width
        y1 = y0 + box_height - 2 * box_margin
        
        # Determine colors based on whether this is the active label
        is_active = (label == current_label)
        
        if is_active:
            # Active: bright colored box
            bg_color = (76, 175, 80)  # Green
            text_color = (255, 255, 255)
            border_color = (56, 142, 60)
        else:
            # Inactive: greyed out
            bg_color = (80, 80, 80)
            text_color = (150, 150, 150)
            border_color = (60, 60, 60)
        
        # Draw box
        draw.rectangle([x0, y0, x1, y1], fill=bg_color, outline=border_color, width=2)
        
        display_label =  label
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), display_label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = x0 + (box_width - text_width) // 2
        text_y = y0 + (box_height - 2 * box_margin - text_height) // 2
        
        draw.text((text_x, text_y), display_label, font=font, fill=text_color)
    
    return np.array(new_img)


def add_labels_to_video(
    input_video_path: str,
    output_video_path: str,
    frame_labels: List[str],
    latent_fps: float = 7.5,
    video_fps: float = 30.0,
    box_height: int = 60,
    max_labels_per_row: int = 6,
) -> bool:
    """
    Add label boxes to a video, creating a new video with labels at the bottom.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        frame_labels: List of labels (one per latent frame)
        latent_fps: FPS of the latent space
        video_fps: FPS of the video
        box_height: Height of label box area
        max_labels_per_row: Max labels per row
    
    Returns:
        True if successful
    """
    try:
        import moviepy.editor as mp
        
        # Load video
        video = mp.VideoFileClip(input_video_path)
        
        # Get unique labels while preserving order of first appearance
        seen = set()
        unique_labels = []
        for label in frame_labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        # Calculate the upsample factor (latent -> video)
        upsample_factor = video_fps / latent_fps
        
        def process_frame(get_frame, t):
            """Process each frame to add label boxes."""
            frame = get_frame(t)
            
            # Calculate which latent frame this corresponds to
            video_frame_idx = int(t * video_fps)
            latent_frame_idx = int(video_frame_idx / upsample_factor)
            latent_frame_idx = min(latent_frame_idx, len(frame_labels) - 1)
            
            if latent_frame_idx >= 0 and latent_frame_idx < len(frame_labels):
                current_label = frame_labels[latent_frame_idx]
            else:
                current_label = ""
            
            # Add label boxes
            frame_with_labels = add_label_boxes_to_frame(
                frame,
                unique_labels,
                current_label,
                box_height=box_height,
                max_labels_per_row=max_labels_per_row,
            )
            
            return frame_with_labels
        
        # Apply the transformation
        processed_video = video.fl(process_frame)
        
        # Write output
        processed_video.write_videofile(
            output_video_path,
            **get_drive_safe_videofile_kwargs(fps=video_fps),
        )
        
        video.close()
        processed_video.close()
        
        return True
        
    except Exception as e:
        logger.warning(f"add_labels_to_video: failed: {e}")
        traceback.print_exc()
        return False


class SampleLoggerActionPlan(Callback):
    """Generate and save samples for ActionPlan latent representations (32-dim).
    
    This callback:
    1. Generates latent samples from the diffusion model
    2. Splits into motion latents (first 16) and CLIP embeddings (last 16)
    3. Decodes motion latents to 272-dim using TAE, then renders as SMPL mesh
    4. Retrieves text labels from CLIP embeddings using inverse PCA + KNN
    5. Creates videos with label boxes showing active labels per frame
    6. Logs comparison between generated and ground truth
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        save_dir: str | None = None,
        guidance_weight: float | None = None,
        save_numpy: bool = True,
        render_video: bool = True,
        include_ground_truth: bool = True,
        # TAE checkpoint for decoding motion latents
        tae_checkpoint: str | None = None,
        # CLIP decoding mode (AE only; non-AE values fallback to AE)
        clip_mode: str = "ae",
        # CLIP text retrieval paths
        autoencoder_path: str | None = None,
        clip_embeddings_path: str | None = None,
        label_id_path: str | None = None,
        # Number of frames to sample for text summary
        text_sample_frames: int = 10,
        # Label box visualization settings
        label_box_height: int = 60,
        max_labels_per_row: int = 6,
    ):
        """
        Args:
            every_n_epochs: Log samples every N epochs
            save_dir: Directory to save samples (uses trainer default if None)
            guidance_weight: CFG weight for sampling
            save_numpy: Save raw numpy files
            render_video: Render motion videos with label boxes
            include_ground_truth: Include ground truth comparison
            tae_checkpoint: Path to TAE checkpoint for decoding (uses default if None)
            clip_mode: CLIP decode mode (AE only; non-AE values fallback to AE)
            autoencoder_path: Path to autoencoder checkpoint
            clip_embeddings_path: Path to full CLIP embeddings (uses default if None)
            label_id_path: Path to label-to-id mapping (uses default if None)
            text_sample_frames: Number of frames to sample for text summary
            label_box_height: Height of label box area in video
            max_labels_per_row: Maximum labels per row in video
        """
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.save_dir = save_dir
        self.guidance_weight = guidance_weight
        self.save_numpy = save_numpy
        self.render_video = render_video
        self.include_ground_truth = include_ground_truth
        self.tae_checkpoint = tae_checkpoint
        self.text_sample_frames = text_sample_frames
        self.label_box_height = label_box_height
        self.max_labels_per_row = max_labels_per_row

        # Initialize text retriever (lazy-loaded)
        self._text_retriever = CLIPTextRetriever(
            mode=clip_mode,
            autoencoder_path=autoencoder_path or DEFAULT_AE_PATH,
            clip_embeddings_path=clip_embeddings_path or DEFAULT_CLIP_EMBEDDINGS_PATH,
            label_id_path=label_id_path or DEFAULT_LABEL_ID_PATH,
        )

        # Lazy-loaded TAE model
        self._tae_model = None

    def _get_tae_model(self, device):
        """Lazy-load the TAE model."""
        if self._tae_model is None:
            try:
                from src.tae import load_tae
                self._tae_model = load_tae(
                    checkpoint_path=self.tae_checkpoint,
                    device=device
                )
                logger.info("SampleLoggerActionPlan: loaded TAE model for decoding")
            except Exception as e:
                logger.warning(f"SampleLoggerActionPlan: failed to load TAE: {e}")
                return None
        return self._tae_model

    def _decode_motion_latent_to_272(self, motion_latent: torch.Tensor, device) -> Optional[np.ndarray]:
        """Decode 16-dim motion latent to 272-dim motion.
        
        Args:
            motion_latent: [T, 16] motion latent features
            device: torch device
            
        Returns:
            [T*4, 272] decoded motion features (4x upsampled)
        """
        try:
            from src.tae import decode_latents
            
            # Decode with denormalization
            decoded = decode_latents(
                motion_latent.numpy(),
                model=self._get_tae_model(device),
                device=device,
                remove_reference_token=False,
                denormalize=True,
            )
            
            return decoded.numpy()
        except Exception as e:
            logger.warning(f"SampleLoggerActionPlan: failed to decode motion latent: {e}")
            traceback.print_exc()
            return None

    def _render_video(
        self,
        motion_272: np.ndarray,
        output_path: str,
        fps: float = 30.0,
        text_overlay: str = "",
        vertex_color: list = None,
    ) -> bool:
        """Render 272-dim motion to video using SMPL."""
        try:
            # Set headless GL backend if no display
            if os.environ.get("DISPLAY") in (None, ""):
                os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
                os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

            from src.renderer.humor import HumorRenderer
            from src.tools.streamer272_feats import streamer272_to_smpl
            from src.tools.smpl_layer import SMPLH

            # Get SMPL model path
            cb_dir = os.path.dirname(os.path.abspath(__file__))
            actionplan_root = os.path.abspath(os.path.join(cb_dir, "..", ".."))
            smplh_path = os.path.join(actionplan_root, "deps", "smplh")

            # Build SMPL layer
            smpl_layer = SMPLH(
                path=smplh_path,
                jointstype="vertices",
                input_pose_rep="axisangle",
                gender="neutral",
                batch_size=512,
            )

            # Convert 272-dim features to SMPL parameters
            motion_tensor = torch.from_numpy(motion_272).float()
            smpl_data = streamer272_to_smpl(motion_tensor)
            poses = smpl_data["poses"]  # [T, 66] axis-angle
            trans = smpl_data["trans"]  # [T, 3]

            # Get vertices from SMPL layer
            vertices = smpl_layer(poses, trans).cpu().numpy()  # [T, 6890, 3]

            # Render mesh video
            # Latent model outputs Y-up, so we need to convert to Z-up for visualization
            renderer = HumorRenderer(fps=fps, convert_yup_to_zup=True)
            render_kwargs = {
                "vertices": vertices,
                "output": output_path,
                "text_overlay": text_overlay,
            }
            if vertex_color is not None:
                render_kwargs["vertex_color"] = vertex_color

            renderer(**render_kwargs)

            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            return file_size > 0

        except Exception as e:
            logger.warning(f"SampleLoggerActionPlan: failed to render video: {e}")
            traceback.print_exc()
            return False

    def _split_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split 32-dim features into motion latents (16) and CLIP embeddings (16).
        
        Args:
            x: [T, 32] combined latent representation
            
        Returns:
            motion_latents: [T, 16] motion latent features
            clip_embeds: [T, 16] PCA-reduced CLIP embeddings
        """
        assert x.shape[-1] == 32, f"Expected 32 features, got {x.shape[-1]}"
        motion_latents = x[..., :16]
        clip_embeds = x[..., 16:]
        return motion_latents, clip_embeds

    def _get_text_summary(self, clip_embeds: np.ndarray, prefix: str = "") -> str:
        """Get a text summary from CLIP embeddings."""
        summary = self._text_retriever.get_summary_text(clip_embeds, self.text_sample_frames)
        if summary and prefix:
            return f"{prefix}: {summary}"
        return summary

    def _get_detailed_texts(
        self,
        clip_embeds: np.ndarray,
        num_samples: int = 10,
    ) -> List[Tuple[int, str]]:
        """Get detailed text labels for sampled frames."""
        T = clip_embeds.shape[0]
        if T == 0:
            return []
        
        sample_indices = np.linspace(0, T - 1, min(num_samples, T), dtype=int).tolist()
        texts, _ = self._text_retriever.retrieve_texts(clip_embeds, sample_indices)
        
        return list(zip(sample_indices, texts))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Only rank 0 generates samples and writes files (for DDP compatibility)
        if not trainer.is_global_zero:
            return
        
        epoch = int(trainer.current_epoch)
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        try:
            dl = None
            # Try val dataloader first
            val_dls = getattr(trainer, "val_dataloaders", None)
            if val_dls is not None:
                if isinstance(val_dls, (list, tuple)):
                    if len(val_dls) > 0:
                        dl = val_dls[0]
                else:
                    dl = val_dls
            
            # Fall back to train dataloader
            if dl is None:
                logger.debug("SampleLoggerActionPlan: Using train dataloader for sample rendering")
                train_dl = getattr(trainer, "train_dataloader", None)
                if train_dl is not None:
                    dl = train_dl
            
            if dl is None:
                logger.debug("SampleLoggerActionPlan: no dataloader available, skipping")
                return

            # Check if dataloader has any data
            try:
                if hasattr(dl, 'dataset') and len(dl.dataset) == 0:
                    logger.debug("SampleLoggerActionPlan: dataset is empty, skipping")
                    return
            except Exception:
                pass

            batch = next(iter(dl))
        except StopIteration:
            logger.debug("SampleLoggerActionPlan: dataloader is empty, skipping sample generation")
            return
        except Exception as e:
            logger.debug(f"SampleLoggerActionPlan: could not fetch batch ({type(e).__name__}), skipping")
            return

        # Select sample from batch
        device = pl_module.device
        try:
            bs = int(batch.get("x").shape[0]) if isinstance(batch.get("x"), torch.Tensor) else 1
        except Exception:
            bs = 1

        trigger_idx = (epoch + 1) // self.every_n_epochs
        sample_idx = int(trigger_idx % max(bs, 1))

        # Get sample metadata
        if isinstance(batch.get("length"), torch.Tensor):
            length = int(batch["length"][sample_idx].item())
        else:
            length = int(batch.get("length", [78])[sample_idx])

        texts = batch.get("text", [""])
        if isinstance(texts, (list, tuple)) and len(texts) > sample_idx:
            text = str(texts[sample_idx])
        else:
            text = str(texts) if not isinstance(texts, torch.Tensor) else ""

        # Prepare text embeddings
        tx = batch.get("tx")
        tx_uncond = batch.get("tx_uncond")
        tx = _move_tx_to_device(tx, device)
        if isinstance(tx, dict):
            tx = _slice_first_dim_if_batched(tx, sample_idx, sample_idx + 1)
        if isinstance(tx_uncond, dict):
            tx_uncond = _move_tx_to_device(tx_uncond, device)
            tx_uncond = _slice_first_dim_if_batched(tx_uncond, sample_idx, sample_idx + 1)
        else:
            tx_uncond = dict(tx) if isinstance(tx, dict) else tx

        # Build infos
        infos = {
            "all_lengths": [length],
            "all_texts": [text],
        }
        if self.guidance_weight is not None:
            infos["guidance_weight"] = float(self.guidance_weight)

        # Ensure lengths are tensors
        if isinstance(tx.get("length"), int):
            tx["length"] = torch.tensor([tx["length"]], device=device)
        if isinstance(tx_uncond.get("length"), int):
            tx_uncond["length"] = torch.tensor([tx_uncond["length"]], device=device)

        # Generate sample
        with torch.no_grad():
            try:
                xstarts = pl_module(tx, tx_uncond, infos, progress_bar=None)
            except TypeError:
                xstarts = pl_module.text_forward(tx, tx_uncond, infos, progress_bar=None)

        if not isinstance(xstarts, torch.Tensor):
            logger.warning("SampleLoggerActionPlan: unexpected output type")
            return

        xstart = xstarts[0, :length].detach().cpu()  # [T, 32]

        # Verify feature dimension
        nfeats = xstart.shape[-1]
        if nfeats != 32:
            logger.warning(f"SampleLoggerActionPlan: expected 32 features, got {nfeats}")
            return

        # Get ground truth if available
        ground_truth = None
        if self.include_ground_truth and "x" in batch:
            try:
                ground_truth = batch["x"][sample_idx, :length].detach().cpu()
            except Exception as e:
                logger.warning(f"SampleLoggerActionPlan: failed to get ground truth: {e}")

        # Split features
        gen_motion, gen_clip = self._split_features(xstart)
        gt_motion, gt_clip = None, None
        if ground_truth is not None:
            gt_motion, gt_clip = self._split_features(ground_truth)

        # Retrieve ALL frame labels from CLIP embeddings (for video visualization)
        gen_frame_labels, _ = self._text_retriever.retrieve_texts_all_frames(gen_clip.numpy())
        gt_frame_labels = []
        if gt_clip is not None:
            gt_frame_labels, _ = self._text_retriever.retrieve_texts_all_frames(gt_clip.numpy())

        # Get text summaries
        gen_text_summary = self._get_text_summary(gen_clip.numpy(), "Generated")
        gt_text_summary = ""
        if gt_clip is not None:
            gt_text_summary = self._get_text_summary(gt_clip.numpy(), "Ground Truth")

        # Get FPS from motion loader
        motion_loader = getattr(getattr(dl, "dataset", None), "motion_loader", None)
        latent_fps = getattr(motion_loader, "fps", 7.5)
        # Decoded 272-dim is at 30 FPS (4x upsampled)
        motion_fps = 30.0

        # Prepare output directory
        base_dir = self.save_dir or getattr(trainer.logger, "save_dir", None) or trainer.default_root_dir
        out_dir = os.path.join(base_dir, "train_samples")
        os.makedirs(out_dir, exist_ok=True)

        # Log results
        logger.info(f"SampleLoggerActionPlan: Epoch {epoch}")
        logger.info(f"  Input text: {text[:100]}...")
        logger.info(f"  Generated length: {length} frames")
        if gen_text_summary:
            logger.info(f"  {gen_text_summary}")
        if gt_text_summary:
            logger.info(f"  {gt_text_summary}")

        # Save numpy files
        if self.save_numpy:
            # Save full 32-dim latents
            np_path = os.path.join(out_dir, f"epoch_{epoch:06d}_generated_actionplan.npy")
            np.save(np_path, xstart.numpy())
            logger.info(f"SampleLoggerActionPlan: saved generated latent to {np_path}")

            # Save split features
            motion_path = os.path.join(out_dir, f"epoch_{epoch:06d}_generated_motion_latent.npy")
            clip_path = os.path.join(out_dir, f"epoch_{epoch:06d}_generated_clip_embed.npy")
            np.save(motion_path, gen_motion.numpy())
            np.save(clip_path, gen_clip.numpy())

            # Log statistics
            logger.info(f"  Motion latent stats - Mean: {gen_motion.mean():.4f}, Std: {gen_motion.std():.4f}")
            logger.info(f"  CLIP embed stats - Mean: {gen_clip.mean():.4f}, Std: {gen_clip.std():.4f}")

            if ground_truth is not None:
                gt_np_path = os.path.join(out_dir, f"epoch_{epoch:06d}_ground_truth_actionplan.npy")
                np.save(gt_np_path, ground_truth.numpy())
                
                # Compute distance between generated and GT
                mse_motion = ((gen_motion - gt_motion) ** 2).mean().item()
                mse_clip = ((gen_clip - gt_clip) ** 2).mean().item()
                logger.info(f"  MSE(gen, gt) motion: {mse_motion:.6f}, clip: {mse_clip:.6f}")

        # Render videos with label visualization
        if self.render_video:
            logger.info("SampleLoggerActionPlan: decoding motion latents to 272-dim...")
            
            # Decode generated motion latent
            gen_decoded_272 = self._decode_motion_latent_to_272(gen_motion, device)
            
            if gen_decoded_272 is not None:
                logger.info(f"  Decoded motion shape: {gen_decoded_272.shape}")
                
                # Save decoded 272-dim numpy
                if self.save_numpy:
                    decoded_np_path = os.path.join(out_dir, f"epoch_{epoch:06d}_generated_272.npy")
                    np.save(decoded_np_path, gen_decoded_272)
                
                # Render generated video (without labels first)
                gen_video_raw_path = os.path.join(out_dir, f"epoch_{epoch:06d}_generated_raw.mp4")
                gen_video_path = os.path.join(out_dir, f"epoch_{epoch:06d}_generated.mp4")
                
                if self._render_video(
                    gen_decoded_272,
                    gen_video_raw_path,
                    fps=motion_fps,
                    text_overlay=f"Generated: {text}",
                    vertex_color=[0.5, 0.0, 0.5],  # Purple for generated
                ):
                    logger.info(f"SampleLoggerActionPlan: rendered raw video to {gen_video_raw_path}")
                    
                    # Add label boxes to video
                    if gen_frame_labels and HAS_PIL:
                        if add_labels_to_video(
                            gen_video_raw_path,
                            gen_video_path,
                            gen_frame_labels,
                            latent_fps=latent_fps,
                            video_fps=motion_fps,
                            box_height=self.label_box_height,
                            max_labels_per_row=self.max_labels_per_row,
                        ):
                            logger.info(f"SampleLoggerActionPlan: added labels, saved to {gen_video_path}")
                            # Clean up raw video
                            try:
                                os.remove(gen_video_raw_path)
                            except Exception:
                                pass
                        else:
                            # Fall back to raw video
                            os.rename(gen_video_raw_path, gen_video_path)
                    else:
                        os.rename(gen_video_raw_path, gen_video_path)
                    
                    # Render ground truth and create comparison
                    if gt_motion is not None:
                        logger.info("SampleLoggerActionPlan: decoding ground truth motion latent...")
                        gt_decoded_272 = self._decode_motion_latent_to_272(gt_motion, device)
                        
                        if gt_decoded_272 is not None:
                            gt_video_raw_path = os.path.join(out_dir, f"epoch_{epoch:06d}_ground_truth_raw.mp4")
                            gt_video_path = os.path.join(out_dir, f"epoch_{epoch:06d}_ground_truth.mp4")
                            
                            if self._render_video(
                                gt_decoded_272,
                                gt_video_raw_path,
                                fps=motion_fps,
                                text_overlay=f"Ground Truth: {text}",
                            ):
                                # Add label boxes to GT video
                                if gt_frame_labels and HAS_PIL:
                                    if add_labels_to_video(
                                        gt_video_raw_path,
                                        gt_video_path,
                                        gt_frame_labels,
                                        latent_fps=latent_fps,
                                        video_fps=motion_fps,
                                        box_height=self.label_box_height,
                                        max_labels_per_row=self.max_labels_per_row,
                                    ):
                                        try:
                                            os.remove(gt_video_raw_path)
                                        except Exception:
                                            pass
                                    else:
                                        os.rename(gt_video_raw_path, gt_video_path)
                                else:
                                    os.rename(gt_video_raw_path, gt_video_path)
                                
                                logger.info(f"SampleLoggerActionPlan: saved ground truth video to {gt_video_path}")
                                
                                # Create side-by-side comparison
                                comparison_path = os.path.join(out_dir, f"epoch_{epoch:06d}_comparison.mp4")
                                if create_side_by_side_video(
                                    gen_video_path,
                                    gt_video_path,
                                    comparison_path,
                                    fps=motion_fps,
                                ):
                                    logger.info(f"SampleLoggerActionPlan: saved comparison video to {comparison_path}")
                                    
                                    # Try to log to wandb
                                    self._log_video_to_wandb(comparison_path, text, motion_fps, trainer)

        # Save detailed text comparison
        metadata_path = os.path.join(out_dir, f"epoch_{epoch:06d}_text_comparison.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Input text prompt: {text}\n")
            f.write(f"Length: {length} frames\n")
            f.write(f"Latent FPS: {latent_fps}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("GENERATED CLIP TEXT RETRIEVAL\n")
            f.write("=" * 60 + "\n")
            gen_detailed = self._get_detailed_texts(gen_clip.numpy(), self.text_sample_frames)
            for frame_idx, txt in gen_detailed:
                f.write(f"  Frame {frame_idx:3d}: {txt}\n")
            f.write(f"\nSummary: {gen_text_summary}\n")
            
            # Count label frequencies
            if gen_frame_labels:
                f.write(f"\nLabel frequency (total {len(gen_frame_labels)} frames):\n")
                label_counts = Counter(gen_frame_labels)
                for label, count in label_counts.most_common():
                    f.write(f"  {label}: {count} frames ({100*count/len(gen_frame_labels):.1f}%)\n")
            
            if gt_clip is not None:
                f.write("\n" + "=" * 60 + "\n")
                f.write("GROUND TRUTH CLIP TEXT RETRIEVAL\n")
                f.write("=" * 60 + "\n")
                gt_detailed = self._get_detailed_texts(gt_clip.numpy(), self.text_sample_frames)
                for frame_idx, txt in gt_detailed:
                    f.write(f"  Frame {frame_idx:3d}: {txt}\n")
                f.write(f"\nSummary: {gt_text_summary}\n")
                
                if gt_frame_labels:
                    f.write(f"\nLabel frequency (total {len(gt_frame_labels)} frames):\n")
                    label_counts = Counter(gt_frame_labels)
                    for label, count in label_counts.most_common():
                        f.write(f"  {label}: {count} frames ({100*count/len(gt_frame_labels):.1f}%)\n")

        logger.info(f"SampleLoggerActionPlan: saved text comparison to {metadata_path}")

        # Try to log text comparison to wandb
        self._log_to_wandb(metadata_path, text, gen_text_summary, gt_text_summary, trainer, epoch)

    def _log_video_to_wandb(self, video_path: str, text: str, fps: float, trainer: Trainer):
        """Try to log video to wandb."""
        if not os.path.exists(video_path):
            return

        try:
            logger_objs = []
            if hasattr(trainer, "loggers") and trainer.loggers:
                logger_objs = list(trainer.loggers)
            elif getattr(trainer, "logger", None) is not None:
                logger_objs = [trainer.logger]

            for lg in logger_objs:
                try:
                    from pytorch_lightning.loggers import WandbLogger
                    import wandb
                    if isinstance(lg, WandbLogger):
                        run = lg.experiment
                        if run is not None and hasattr(run, "log"):
                            video = wandb.Video(video_path, fps=fps, caption=text[:100], format="mp4")
                            run.log({"train/sample_video_actionplan": video})
                            logger.info("SampleLoggerActionPlan: logged video to wandb")
                            return
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"SampleLoggerActionPlan: wandb video upload failed: {e}")

    def _log_to_wandb(
        self,
        metadata_path: str,
        input_text: str,
        gen_text_summary: str,
        gt_text_summary: str,
        trainer: Trainer,
        epoch: int,
    ):
        """Try to log text comparison to wandb."""
        try:
            logger_objs = []
            if hasattr(trainer, "loggers") and trainer.loggers:
                logger_objs = list(trainer.loggers)
            elif getattr(trainer, "logger", None) is not None:
                logger_objs = [trainer.logger]

            for lg in logger_objs:
                try:
                    from pytorch_lightning.loggers import WandbLogger
                    import wandb
                    if isinstance(lg, WandbLogger):
                        run = lg.experiment
                        if run is not None and hasattr(run, "log"):
                            # Create a text table for comparison
                            table = wandb.Table(columns=["Type", "Text"])
                            table.add_data("Input Prompt", input_text)
                            if gen_text_summary:
                                table.add_data("Generated CLIP", gen_text_summary)
                            if gt_text_summary:
                                table.add_data("Ground Truth CLIP", gt_text_summary)
                            
                            run.log({
                                "train/actionplan_text_comparison": table,
                                "epoch": epoch,
                            })
                            logger.info("SampleLoggerActionPlan: logged text comparison to wandb")
                            return
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"SampleLoggerActionPlan: wandb upload failed: {e}")
