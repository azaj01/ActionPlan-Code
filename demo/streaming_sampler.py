"""
Streaming ActionPlan Sampler for real-time text-to-motion generation.

Extends the standard ActionPlan sampler with:
- Generator-based sampling that yields blocks as they complete
- Continuation support for conditioning on previous motion frames
- Session state management for multi-prompt streaming
- MP4 rendering for generated motions
"""

import os
import re
import time
import logging
from typing import Optional, Dict, Any, Generator, List
from dataclasses import dataclass, field

import numpy as np
import torch
from omegaconf import DictConfig

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.actionplan_rectified_flow import MOTION_DIM
from src.model.utils import masked
from src.sampler.actionplan_sampler import ActionPlanSampler
from src.tae import decode_latents, load_tae
from src.tools.streamer272_feats import streamer272_to_smpl
from src.tools.smpl_layer import SMPLH
from src.renderer.humor import HumorRenderer
from mesh_utils import vertices_to_base64, compress_vertices_binary

logger = logging.getLogger(__name__)


def _sanitize_filename(text: str, max_len: int = 64) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\.]+", "", text)
    if not text:
        text = "sample"
    return text[:max_len]


def _actionplan_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_headless_gl():
    if os.environ.get("DISPLAY") in (None, ""):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")


@dataclass
class StreamingSession:
    """Maintains state for a streaming generation session."""
    session_id: str
    accumulated_latents: Optional[np.ndarray] = None
    accumulated_vertices: Optional[np.ndarray] = None
    total_frames: int = 0
    prompts: List[str] = field(default_factory=list)
    segment_boundaries: List[tuple] = field(default_factory=list)  # (prompt, start_latent, end_latent)


class StreamingARBlockSampler:
    """
    ActionPlan block streaming sampler for web demos.
    
    Features:
    - Yields completed blocks as generators for SSE streaming
    - Supports continuation from previous motion (conditioning)
    - Manages session state for multi-prompt generation
    """

    LATENT_FPS = 7.5
    DECODED_FPS = 30.0

    def __init__(
        self,
        run_dir: str,
        ckpt_path: Optional[str] = None,
        tae_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        sampling_timesteps: Optional[int] = None,
        num_blocks: int = 10,
        guidance_weight: float = 3.0,
        overlap_frames: int = 8,
        steps_per_block: int = 2,
    ) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _ensure_headless_gl()

        self.run_dir = os.path.abspath(run_dir)
        self.guidance_weight = float(guidance_weight)
        self.overlap_frames = int(overlap_frames)
        self.num_blocks = int(num_blocks)
        self.steps_per_block = max(1, int(steps_per_block))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        self.latent_sampler = ActionPlanSampler(
            run_dir=self.run_dir,
            ckpt_path=ckpt_path,
            device=str(self.device),
            guidance_weight=self.guidance_weight,
            abs_root=False,
            mode="streaming_block",
            num_blocks=self.num_blocks,
            steps_per_block=self.steps_per_block,
            pick_lowest_variance=True,
            sampling_timesteps=sampling_timesteps,
        )
        self.device = self.latent_sampler.device
        self.cfg: DictConfig = self.latent_sampler.cfg
        self.fps = float(self.latent_sampler.fps)
        self.featsname = str(self.latent_sampler.featsname)
        self.ckpt_path = self.latent_sampler.ckpt_path
        self.diffusion = self.latent_sampler.diffusion
        self.text_model = self.latent_sampler.text_model

        actionplan_root = _actionplan_root()
        smplh_path = os.path.join(actionplan_root, "deps", "smplh")
        self.smpl_layer = SMPLH(
            path=smplh_path,
            jointstype="vertices",
            input_pose_rep="axisangle",
            gender="neutral",
            batch_size=512,
        ).to(self.device)

        self.tae_model = load_tae(checkpoint_path=tae_checkpoint, device=self.device)

        self.renderer = HumorRenderer(fps=self.DECODED_FPS)

        self.model = self.diffusion
        self.T = int(getattr(self.diffusion, "timesteps", self.diffusion.motion_steps))
        self.sampling_timesteps = int(self.diffusion.motion_steps)
        self._uses_rectified_flow = bool(getattr(self.diffusion, "uses_rectified_flow", False))
        if self._uses_rectified_flow:
            self.level_to_t_index = torch.linspace(0, 1, steps=self.sampling_timesteps + 1)
        else:
            t_zero_clean = getattr(self.diffusion, "t_zero_clean", False)
            max_t = self.T if t_zero_clean else self.T - 1
            self.level_to_t_index = torch.linspace(0, max_t, steps=self.sampling_timesteps + 1)

        self.sessions: Dict[str, StreamingSession] = {}

    def _resolve_duration(self, length: int) -> tuple[int, int]:
        """Resolve duration (tensor length for denoiser) and effective_length (actual motion frames).
        
        Matches training: pad to inference_length (e.g. 78) when set.
        """
        inference_length = getattr(self.diffusion, "inference_length", None)
        if inference_length is not None:
            duration = int(inference_length)
            effective_length = min(length, duration)
        else:
            duration = length
            effective_length = length
        return duration, effective_length

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpt_dir = os.path.join(self.run_dir, "logs", "checkpoints")
        if not os.path.isdir(ckpt_dir):
            return None
        candidates = []
        for name in os.listdir(ckpt_dir):
            if name.endswith(".ckpt"):
                path = os.path.join(ckpt_dir, name)
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    mtime = 0
                candidates.append((mtime, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _build_text_embeddings(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        tx_emb = self.text_model(texts)
        if isinstance(tx_emb, torch.Tensor):
            tx = {
                "x": tx_emb[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb))], device=self.device),
            }
        else:
            tx = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in tx_emb.items()}
        return tx

    def _decode_latents_to_272(self, latents: np.ndarray) -> np.ndarray:
        """Decode 16-dim latents to 272-dim motion features (without SMPL conversion)."""
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

    def _motion272_to_vertices(self, motion_272: np.ndarray) -> np.ndarray:
        """Convert 272-dim motion to SMPL vertices."""
        motion_tensor = torch.from_numpy(motion_272).float().to(self.device)
        smpl_data = streamer272_to_smpl(motion_tensor)
        poses = smpl_data["poses"].to(self.device)
        trans = smpl_data["trans"].to(self.device)
        vertices = self.smpl_layer(poses, trans).cpu().numpy()
        return vertices

    TAE_FACTOR = 4  # Latent temporal compression: 1 latent = 4 decoded frames

    def get_or_create_session(self, session_id: str) -> StreamingSession:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = StreamingSession(session_id=session_id)
        return self.sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Clear a session's state."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def trim_session(self, session_id: str, keep_frames: int) -> tuple[bool, str]:
        """
        Trim session to keep only the first keep_frames decoded frames.
        Used when the user removes the latest generation(s) from the sequence.
        Returns (success, message).
        """
        if session_id not in self.sessions:
            return False, "session_not_found"
        session = self.sessions[session_id]
        if session.accumulated_vertices is None or session.accumulated_latents is None:
            self.clear_session(session_id)
            return True, "cleared"
        if keep_frames <= 0:
            self.clear_session(session_id)
            return True, "cleared"
        n_verts = session.accumulated_vertices.shape[0]
        if keep_frames >= n_verts:
            return True, "no_op"
        TAE_FACTOR = 4
        keep_latents = keep_frames // TAE_FACTOR
        if keep_latents <= 0:
            self.clear_session(session_id)
            return True, "cleared"
        session.accumulated_vertices = session.accumulated_vertices[:keep_frames]
        session.accumulated_latents = session.accumulated_latents[:keep_latents]
        session.total_frames = keep_latents
        session.prompts = []
        new_boundaries = []
        for p, s, e in session.segment_boundaries:
            if e <= keep_latents:
                new_boundaries.append((p, s, e))
            elif s < keep_latents:
                new_boundaries.append((p, s, keep_latents))
        session.segment_boundaries = new_boundaries
        return True, "trimmed"

    @torch.no_grad()
    def sample_streaming(
        self,
        text: str,
        seconds: float = 5.0,
        session_id: Optional[str] = None,
        num_blocks: Optional[int] = None,
        vertices_format: str = "base64",
        include_final_vertices: bool = True,
        render_video: bool = False,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generate motion with streaming block output.
        
        Yields dictionaries with block data as each block completes.
        If session_id is provided, conditions on previous session motion.
        
        Args:
            text: Text prompt for motion generation
            seconds: Duration in seconds
            session_id: Optional session ID for continuation
            num_blocks: Number of blocks to divide generation into
            vertices_format: Vertex payload format: "base64" or "binary"
            include_final_vertices: Whether to include full motion vertices in
                the final generation_complete event.
            render_video: If True, render MP4 video at the end
            output_dir: Directory for video output
            output_name: Base name for video output
        """
        if vertices_format not in {"base64", "binary"}:
            raise ValueError(f"Unsupported vertices_format: {vertices_format}")
        length = int(round(self.fps * float(seconds)))
        req_num_blocks = int(num_blocks) if num_blocks is not None else self.num_blocks

        session = None
        conditioning_frames = 0
        previous_latents = None

        if session_id:
            session = self.get_or_create_session(session_id)
            if session.accumulated_latents is not None and len(session.accumulated_latents) > 0:
                overlap = min(self.overlap_frames, len(session.accumulated_latents))
                previous_latents = session.accumulated_latents[-overlap:]
                conditioning_frames = overlap
                logger.info(f"Conditioning on {conditioning_frames} frames from session {session_id}")

        total_requested = conditioning_frames + length
        duration, effective_length = self._resolve_duration(total_requested)

        if effective_length < total_requested:
            logger.warning(
                "Requested length %s exceeds inference_length %s; truncating to %s.",
                total_requested, duration, effective_length
            )
            length = effective_length - conditioning_frames

        if req_num_blocks > length:
            req_num_blocks = max(1, length)
        block_size = max(1, (effective_length + req_num_blocks - 1) // req_num_blocks)
        chunk_indices = list(range(0, effective_length, block_size))

        TAE_FACTOR = 4
        conditioning_decoded_frames = conditioning_frames * TAE_FACTOR
        session_decoded_frames = (
            len(session.accumulated_latents) * TAE_FACTOR
            if session and session.accumulated_latents is not None
            else 0
        )

        yield {
            "type": "generation_start",
            "total_blocks": len(chunk_indices),
            "total_new_frames": length * TAE_FACTOR,
            "conditioning_frames": conditioning_frames,
            "duration": duration,
            "effective_length": effective_length,
            "fps": self.DECODED_FPS,
        }

        prev_decoded_frames = session_decoded_frames if session_decoded_frames > 0 else conditioning_decoded_frames
        final_latents = None
        try:
            stream = self.latent_sampler.sample_streaming(
                text=text,
                seconds=seconds,
                fps=self.fps,
                conditioning_latents=previous_latents,
                conditioning_frames=conditioning_frames,
                num_blocks=req_num_blocks,
                progress_bar=None,
            )
            for event in stream:
                event_type = event.get("type")
                if event_type == "block_ready":
                    cumulative_latents = np.asarray(event["latents"], dtype=np.float32)
                    block_i = int(event["block_idx"])
                    if session_decoded_frames > 0 and session is not None and session.accumulated_latents is not None:
                        all_latents_full = np.concatenate(
                            [session.accumulated_latents, cumulative_latents[conditioning_frames:]],
                            axis=0,
                        )
                    else:
                        all_latents_full = cumulative_latents

                    all_decoded = self._decode_latents_to_272(all_latents_full)
                    new_decoded_start = prev_decoded_frames
                    new_decoded_end = all_decoded.shape[0]
                    if new_decoded_end <= new_decoded_start:
                        continue
                    cumulative_vertices = self._motion272_to_vertices(all_decoded[:new_decoded_end])
                    new_vertices = cumulative_vertices[new_decoded_start:new_decoded_end]

                    decoded_start_for_frontend = new_decoded_start - session_decoded_frames
                    decoded_end_for_frontend = new_decoded_end - session_decoded_frames
                    prev_decoded_frames = new_decoded_end

                    chunk_start = chunk_indices[block_i]
                    chunk_end = chunk_indices[block_i + 1] if block_i + 1 < len(chunk_indices) else effective_length
                    block_event = {
                        "type": "block_complete",
                        "block_idx": block_i,
                        "total_blocks": len(chunk_indices),
                        "block_start_frame": decoded_start_for_frontend,
                        "block_end_frame": decoded_end_for_frontend,
                        "latent_frames": chunk_end - chunk_start,
                        "decoded_frames": new_vertices.shape[0],
                        "fps": self.DECODED_FPS,
                    }
                    if vertices_format == "binary":
                        block_event["vertices_bin"] = compress_vertices_binary(new_vertices)
                    else:
                        block_event["vertices_b64"] = vertices_to_base64(new_vertices)
                    yield block_event
                elif event_type == "complete":
                    final_latents = np.asarray(event["latents"], dtype=np.float32)
        finally:
            pass

        if final_latents is None:
            raise RuntimeError("ActionPlan stream did not produce final latents")

        new_latents = final_latents[conditioning_frames:effective_length]

        # Build full sequence: session (from previous runs) + new part from current run
        if session_decoded_frames > 0 and session is not None and session.accumulated_latents is not None:
            all_latents_full = np.concatenate(
                [session.accumulated_latents, new_latents], axis=0
            )
            all_decoded_272 = self._decode_latents_to_272(all_latents_full)
        else:
            all_latents = final_latents[:effective_length]
            all_decoded_272 = self._decode_latents_to_272(all_latents)

        all_vertices = self._motion272_to_vertices(all_decoded_272)
        
        if session:
            start_latent = 0 if session.accumulated_latents is None else len(session.accumulated_latents)
            if session.accumulated_latents is None:
                session.accumulated_latents = new_latents
                session.accumulated_vertices = all_vertices
            else:
                session.accumulated_latents = np.concatenate([
                    session.accumulated_latents, new_latents
                ], axis=0)
                # all_vertices already contains full sequence (prev + new), so replace
                session.accumulated_vertices = all_vertices
            end_latent = len(session.accumulated_latents)
            session.total_frames += new_latents.shape[0]
            session.prompts.append(text)
            session.segment_boundaries.append((text, start_latent, end_latent))
        
        video_path = None
        if render_video:
            video_path = self._render_video(all_vertices, text, output_dir, output_name)
        
        complete_event = {
            "type": "generation_complete",
            "total_latent_frames": all_decoded_272.shape[0],
            "total_decoded_frames": all_vertices.shape[0],
            "session_total_frames": session.total_frames if session else all_vertices.shape[0],
            "video_path": video_path,
        }
        if include_final_vertices:
            if vertices_format == "binary":
                complete_event["vertices_bin"] = compress_vertices_binary(all_vertices)
            else:
                complete_event["vertices_b64"] = vertices_to_base64(all_vertices)
        yield complete_event

    def get_smpl_faces(self) -> np.ndarray:
        """Get SMPL mesh faces for frontend rendering."""
        return self.smpl_layer.smplh.faces.astype(np.int32)

    def get_session_vertices(self, session_id: str) -> Optional[np.ndarray]:
        """Get all accumulated vertices for a session."""
        if session_id in self.sessions:
            return self.sessions[session_id].accumulated_vertices
        return None

    def _render_video(
        self,
        vertices: np.ndarray,
        text: str,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> str:
        """Render vertices to MP4 video."""
        if isinstance(vertices, torch.Tensor):
            vertices = vertices.detach().cpu().numpy()
        
        ts = time.strftime("%Y%m%d-%H%M%S")
        base_name = output_name or f"{_sanitize_filename(text)}_{ts}"
        out_dir = output_dir or os.path.join(self.run_dir, "generations", "streaming")
        os.makedirs(out_dir, exist_ok=True)
        
        video_path = os.path.join(out_dir, f"{base_name}.mp4")
        verts_npy = os.path.join(out_dir, f"{base_name}_verts.npy")
        
        try:
            np.save(verts_npy, vertices)
            logger.info(f"Saved vertices to {verts_npy}")
        except Exception as e:
            logger.warning("Failed to save vertices: %s", e)
        
        try:
            self.renderer(
                vertices=vertices,
                output=video_path,
                text_overlay=f"Generated (streaming): {text}",
            )
            logger.info(f"Rendered video to {video_path}")
        except Exception as e:
            logger.error("Rendering failed: %s", e)
            raise
        
        return video_path

    @torch.no_grad()
    def sample_and_render(
        self,
        text: str,
        seconds: float = 5.0,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate motion and render to video (non-streaming convenience method).
        
        Returns dict with video_path and generation stats.
        """
        result = {}
        for event in self.sample_streaming(
            text=text,
            seconds=seconds,
            session_id=None,
            num_blocks=self.num_blocks,
            render_video=True,
            output_dir=output_dir,
            output_name=output_name,
        ):
            if event["type"] == "generation_start":
                result["total_blocks"] = event["total_blocks"]
                result["duration"] = event.get("duration")
                result["effective_length"] = event.get("effective_length")
            elif event["type"] == "generation_complete":
                result["total_latent_frames"] = event["total_latent_frames"]
                result["total_decoded_frames"] = event["total_decoded_frames"]
                result["video_path"] = event.get("video_path")
        
        return result

    def render_session(
        self,
        session_id: str,
        output_dir: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> Optional[str]:
        """Render all accumulated motion from a session to video."""
        session = self.sessions.get(session_id)
        if session is None or session.accumulated_vertices is None:
            logger.warning(f"No session or vertices for session {session_id}")
            return None
        
        text = " → ".join(session.prompts) if session.prompts else "session_motion"
        return self._render_video(
            session.accumulated_vertices,
            text,
            output_dir,
            output_name or f"session_{session_id}",
        )
