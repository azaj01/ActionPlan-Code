#!/usr/bin/env python3
"""
Unified motion generation script.

Two modes:
  Offline mode (default): random with configurable steps-per-latent (default 2)
  Streaming mode (--streaming): Interactive, text+first motion together; each new segment
    conditioned on previous.
Usage:
    python generate.py                    # 1 random test prompt
    python generate.py -n 10              # 10 random test prompts
    python generate.py --text "a person walks"  # Custom text prompt
    python generate.py --streaming        # Interactive streaming mode (/q quit, /rq render and quit)
    python generate.py --streaming --g1   # Same + ZMQ v3 stream for Sonic / Unitree G1
"""

import argparse
import json
import os
import re
import sys
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import numpy as np
import zmq
from src.sampler import ActionPlanSampler
from src.tools.latent_utils import (
    LatentPipeline,
    LATENT_FPS,
    DECODED_FPS,
)
from src.tools.streamer272_feats import streamer272_to_smpl

# --- Fixed defaults ---
RUN_DIR = "outputs/actionplan"
CKPT = "latest-epoch=9999.ckpt"
USE_GT_DURATION = True
GUIDANCE = 5.5
SECONDS = 10
STREAMING_SECONDS = 5
SPLIT = "test"
SEED = 420
# Annotation dataset for loading test prompts (avoids heavy dataset instantiation)
ANNOTATION_DATASET = "humanml3d_actionplan_merged"
RENDER_MP4 = True
CONDITIONING_FRAMES = 8  # Overlap frames for conditioning between streaming segments
STREAMING_SAMPLING_TIMESTEPS = 10  # Diffusion steps for streaming (motion_steps / text_steps)


def _sanitize_filename(text: str, max_len: int = 64) -> str:
    """Sanitize text for use as filename."""
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\.]+", "", text)
    if not text:
        text = "sample"
    return text[:max_len]


def _select_random_test_prompts(
    *, split: str, k: int, seed: int, return_durations: bool = False, return_motion_ids: bool = False
) -> List[str] | Tuple[List[str], List[float]] | Tuple[List[str], List[float], List[str]]:
    """Load k random test prompts from annotation dataset (no heavy dataset instantiation)."""
    from src.data.text_motion import load_annotations, read_split

    actionplan_root = os.path.abspath(os.path.dirname(__file__))
    annotation_path = os.path.join(actionplan_root, "datasets", "annotations", ANNOTATION_DATASET)
    annotations = load_annotations(annotation_path)

    split_file_path = os.path.join(annotation_path, "splits", f"{split}.txt")
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Split file not found: {split_file_path}")
    keyids = read_split(annotation_path, split)
    keyids = [keyid for keyid in keyids if keyid in annotations]

    if len(keyids) == 0:
        raise RuntimeError(f"No items found for split '{split}' in dataset '{ANNOTATION_DATASET}'")

    rng = random.Random(seed)
    chosen = keyids if k >= len(keyids) else rng.sample(keyids, k)

    prompts: List[str] = []
    durations: List[float] = []
    motion_ids: List[str] = []
    for keyid in chosen:
        ann = annotations[keyid]["annotations"][0]
        prompts.append(str(ann["text"]))
        motion_ids.append(keyid)
        # Calculate duration from annotation start/end times
        duration = ann.get("end", 10.0) - ann.get("start", 0.0)
        durations.append(float(duration))

    if return_motion_ids:
        if return_durations:
            return prompts, durations, motion_ids
        return prompts, motion_ids
    if return_durations:
        return prompts, durations
    return prompts


def _get_device() -> str:
    """Auto-detect best device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_ckpt(run_dir: str, ckpt: str) -> str:
    """Resolve checkpoint path from 'last', 'best', filename, or full path."""
    if ckpt in ("last", "best"):
        return os.path.join(run_dir, "logs", "checkpoints", f"{ckpt}.ckpt")
    if os.path.sep not in ckpt and not os.path.isabs(ckpt):
        return os.path.join(run_dir, "logs", "checkpoints", ckpt)
    return ckpt


def _build_pyramid_sampler(
    run_dir: str, ckpt_path: str, device: str, guidance: float, steps_per_block: int = 2
) -> ActionPlanSampler:
    """Build ActionPlanSampler for pyramid_random (batch mode)."""
    return ActionPlanSampler(
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        device=device,
        guidance_weight=guidance,
        mode="actionplan",
        steps_per_block=steps_per_block,
    )


def _build_streaming_sampler(run_dir: str, ckpt_path: str, device: str, guidance: float) -> ActionPlanSampler:
    """Build ActionPlanSampler for streaming mode (interactive)."""
    return ActionPlanSampler(
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        device=device,
        guidance_weight=guidance,
        mode="streaming",
        steps_per_block=2,
        sampling_timesteps=int(STREAMING_SAMPLING_TIMESTEPS),
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate motion from text prompts using ActionPlan sampler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py
  python generate.py -n 10
  python generate.py --text "a person walks forward"
  python generate.py --streaming        # Interactive mode (/q quit, /rq render and quit)
  python generate.py --streaming --g1   # + ZMQ pose stream (see README)
        """,
    )
    parser.add_argument(
        "-t", "--text",
        type=str,
        default=None,
        help="Custom text prompt (if not set, sample from test set)",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=1,
        help="Number of random prompts from test set (default: 1)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Interactive streaming mode: enter prompts one by one, each conditioned on previous",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Disable timestamped session folder (streaming mode only)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Duration in seconds per segment (default: 10 batch, 5 streaming)",
    )
    parser.add_argument(
        "--steps-per-block",
        type=int,
        default=2,
        help="Pyramid steps per block for batch mode (default: 2)",
    )
    parser.add_argument(
        "--g1",
        action="store_true",
        help="Streaming only: publish SMPL as Sonic ZMQ protocol v3 (requires --streaming)",
    )
    parser.add_argument(
        "--g1-host",
        type=str,
        default="*",
        help="ZMQ PUB bind address (default: * all interfaces)",
    )
    parser.add_argument(
        "--g1-port",
        type=int,
        default=5556,
        help="ZMQ PUB port (default: 5556)",
    )
    parser.add_argument(
        "--g1-topic",
        type=str,
        default="pose",
        help="ZMQ topic prefix for v3 packets (default: pose)",
    )
    parser.add_argument(
        "--g1-hz",
        type=float,
        default=30.0,
        help="ZMQ publish rate in Hz (default: 30)",
    )
    return parser.parse_args()


def _run_streaming_mode(args, run_dir: str, ckpt_path: str, device: str) -> None:
    """Interactive streaming mode: enter prompts one by one, conditioned on previous."""
    out_root = os.path.join(run_dir, "generations", "generate_streaming_latent")
    os.makedirs(out_root, exist_ok=True)

    if getattr(args, "no_timestamp", False):
        session_dir = out_root
    else:
        session_dir = os.path.join(out_root, "session_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(session_dir, exist_ok=True)

    g1_pub = None
    g1_state = None
    if getattr(args, "g1", False):
        from src.tools.g1_zmq_adapter import (
            G1ZmqFrameState,
            create_g1_publisher,
            expand_poses_for_zmq_queue,
        )

        g1_state = G1ZmqFrameState()
        try:
            g1_pub = create_g1_publisher(
                host=args.g1_host,
                port=args.g1_port,
                topic=args.g1_topic,
                hz=float(args.g1_hz),
            )
        except Exception as e:
            print(f"[g1-zmq] Failed to initialize publisher: {e}")
            sys.exit(1)

    latent_pipeline = LatentPipeline(None, device=device)
    sampler = _build_streaming_sampler(run_dir, ckpt_path, device, float(GUIDANCE))
    conditioning_frames = max(0, int(CONDITIONING_FRAMES))

    print("=" * 60)
    print("STREAMING MODE (interactive)")
    print("  Session directory:", session_dir)
    print("  Conditioning:", conditioning_frames, "frames overlap between segments")
    if g1_pub is not None:
        print(
            f"  G1 / Sonic ZMQ: tcp://{args.g1_host}:{args.g1_port} topic={args.g1_topic!r} @ {args.g1_hz} Hz"
        )
    print()
    print("  Output layout (SMPL params written as each frame is denoised):")
    print("    - Per segment: {session}/{prompt}/")
    print("        streaming_smpl_current.npz  # live SMPL (poses, trans, joints) during denoise")
    print("        {prompt}_latents.npy        # final latent features")
    print("        {prompt}_decoded272.npy     # final 272-d motion")
    print("        {prompt}.npz                # final SMPL params")
    print("    - Full session (multi-segment): {session}/streaming_session/")
    print("        streaming_session_latents.npy")
    print("        streaming_session_decoded272.npy")
    print()
    print("  Commands: /q = quit, /rq = render full sequence and quit, /reset = new session")
    print("=" * 60)

    accumulated_latents: Optional[np.ndarray] = None
    segment_lengths: List[int] = []
    prompts: List[str] = []
    gen_count = 0
    render_on_exit = False

    try:
        while True:
            try:
                text = input("\nText prompt> ").strip()
            except EOFError:
                print("\nExiting.")
                break

            if not text:
                continue
            if text.lower() == "/q":
                print("Exiting.")
                break
            if text.lower() == "/rq":
                render_on_exit = True
                print("Will render full sequence and quit.")
                break
            if text.lower() == "/reset":
                accumulated_latents = None
                segment_lengths.clear()
                prompts.clear()
                if g1_state is not None:
                    g1_state.full_reset()
                print("  Session reset. Next prompt will start a new sequence.")
                continue

            gen_count += 1
            seed = SEED + gen_count - 1
            cond_latents = None
            cond_frames = 0
            if accumulated_latents is not None and conditioning_frames > 0:
                cond_latents = accumulated_latents[-conditioning_frames:]
                cond_frames = conditioning_frames
                print(f"\n[{gen_count}] Generating: \"{text}\" ({args.seconds}s) [conditioned on previous]...")
            else:
                print(f"\n[{gen_count}] Generating: \"{text}\" ({args.seconds}s)...")

            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            safe = _sanitize_filename(text)
            prompt_root = os.path.join(session_dir, safe)
            os.makedirs(prompt_root, exist_ok=True)
            smpl_stream_path = os.path.join(prompt_root, "streaming_smpl_current.npz")

            try:
                stream = sampler.sample_streaming(
                    text=text,
                    seconds=float(args.seconds),
                    fps=LATENT_FPS,
                    conditioning_latents=cond_latents,
                    conditioning_frames=cond_frames,
                    progress_bar=None,
                )
            except Exception as e:
                print(f"  Error: {e}")
                continue

            if g1_state is not None:
                g1_state.reset_for_new_prompt(cond_frames)

            final_latents = None
            for event in stream:
                if event["type"] == "frame_ready":
                    latents = event["latents"]
                    if latents.size > 0:
                        motion_272 = latent_pipeline.decode_latents(latents)
                        smpl_data = streamer272_to_smpl(torch.from_numpy(motion_272).float())
                        poses_np = smpl_data["poses"].cpu().numpy()
                        trans_np = smpl_data["trans"].cpu().numpy()
                        np.savez(
                            smpl_stream_path,
                            poses=poses_np,
                            trans=trans_np,
                            joints=smpl_data["joints"].cpu().numpy(),
                            fps=np.array(DECODED_FPS),
                        )
                        if g1_pub is not None and g1_state is not None:
                            for p66, t3, fidx in expand_poses_for_zmq_queue(poses_np, trans_np, g1_state):
                                g1_pub.enqueue(p66, t3, fidx)
                elif event["type"] == "complete":
                    final_latents = event["latents"]

            if final_latents is None:
                continue

            if g1_state is not None:
                g1_state.finalize_prompt()

            segment_lengths.append(len(final_latents))
            if accumulated_latents is None:
                accumulated_latents = final_latents.copy()
            else:
                accumulated_latents = np.concatenate(
                    [accumulated_latents, final_latents[conditioning_frames:]], axis=0
                )
            prompts.append(text)

            outputs = latent_pipeline.decode_and_render(
                final_latents,
                Path(prompt_root),
                safe,
                text,
                render=False,
            )
            motion_272 = latent_pipeline.decode_latents(final_latents)
            smpl_data = streamer272_to_smpl(torch.from_numpy(motion_272).float())
            npz_path = os.path.join(prompt_root, f"{safe}.npz")
            np.savez(
                npz_path,
                poses=smpl_data["poses"].cpu().numpy(),
                trans=smpl_data["trans"].cpu().numpy(),
                joints=smpl_data["joints"].cpu().numpy(),
                fps=np.array(DECODED_FPS),
                text=np.array(text),
            )
            print(f"  Saved: {outputs['decoded_path']}, {npz_path}")
            print("  Done.")

        if render_on_exit and accumulated_latents is None:
            print("Nothing to render.")

        if accumulated_latents is not None:
            motion_272 = latent_pipeline.decode_latents(accumulated_latents)
            if len(prompts) > 1:
                session_name = "streaming_session"
                session_root = Path(session_dir) / session_name
                session_root.mkdir(parents=True, exist_ok=True)
                np.save(session_root / f"{session_name}_latents.npy", accumulated_latents)
                np.save(session_root / f"{session_name}_decoded272.npy", motion_272)
                print(f"\nSaved full conditioned sequence: {session_root}/")

            if render_on_exit:
                mp4_path = os.path.join(session_dir, "streaming_session.mp4")
                n_decoded = motion_272.shape[0]
                n_latent = accumulated_latents.shape[0]
                text_overlay_seq = [""] * n_decoded
                if len(prompts) > 0 and n_latent > 0:
                    cum = 0
                    for i, (L, prompt) in enumerate(zip(segment_lengths, prompts)):
                        if i == 0:
                            start_lat, end_lat = 0, L
                        else:
                            start_lat = cum - conditioning_frames
                            end_lat = cum + L - conditioning_frames
                        start_dec = int(round(start_lat * n_decoded / n_latent))
                        end_dec = int(round(end_lat * n_decoded / n_latent))
                        start_dec = max(0, min(start_dec, n_decoded))
                        end_dec = max(start_dec, min(end_dec, n_decoded))
                        for j in range(start_dec, end_dec):
                            text_overlay_seq[j] = prompt
                        cum = end_lat
                latent_pipeline.render_motion(
                    motion_272,
                    output_path=mp4_path,
                    text_overlay="",
                    text_overlay_seq=text_overlay_seq,
                )
                print(f"Rendered full sequence: {mp4_path}")

        print(f"\nGenerated {gen_count} motion(s). Outputs in: {session_dir}")
    finally:
        if g1_pub is not None:
            g1_pub.stop()


def main():
    args = _parse_args()
    if args.seconds is None:
        args.seconds = STREAMING_SECONDS if args.streaming else SECONDS

    if getattr(args, "g1", False) and not args.streaming:
        print("Error: --g1 requires --streaming", file=sys.stderr)
        sys.exit(2)

    if args.streaming:
        run_dir = os.path.abspath(RUN_DIR)
        ckpt_path = _resolve_ckpt(run_dir, CKPT)
        device = _get_device()
        _run_streaming_mode(args, run_dir, ckpt_path, device)
        return

    # Batch mode (pyramid_random_s2)
    num_samples = args.num_samples
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("generate")

    run_dir = os.path.abspath(RUN_DIR)
    ckpt_path = _resolve_ckpt(run_dir, CKPT)
    device = _get_device()
    latent_pipeline = LatentPipeline(None, device=device)

    out_root = os.path.join(run_dir, "generations", "actionplan")
    os.makedirs(out_root, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GENERATION CONFIGURATION (ActionPlan)")
    logger.info("=" * 60)
    if args.text is not None:
        logger.info("Prompt: custom text (%s)", args.text[:50] + "..." if len(args.text) > 50 else args.text)
    else:
        logger.info("Num samples: %d (from test set)", num_samples)
    logger.info("Run directory: %s", run_dir)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Output root: %s", out_root)
    logger.info("Device: %s", device)
    logger.info("Latent FPS: %.2f -> Render FPS: %.1f", LATENT_FPS, DECODED_FPS)
    logger.info("=" * 60)

    if args.text is not None:
        prompts = [args.text.strip()]
        durations = [float(args.seconds)]
        motion_ids = None
    else:
        # Sample random prompts from test set
        if USE_GT_DURATION:
            prompts, durations, motion_ids = _select_random_test_prompts(
                split=str(SPLIT), k=num_samples, seed=int(SEED),
                return_durations=True, return_motion_ids=True,
            )
        else:
            prompts, motion_ids = _select_random_test_prompts(
                split=str(SPLIT), k=num_samples, seed=int(SEED),
                return_durations=False, return_motion_ids=True,
            )
            durations = [float(SECONDS)] * len(prompts)

    prompts_path = os.path.join(out_root, "prompts_used.txt")
    with open(prompts_path, "w", encoding="utf-8") as f:
        f.write(f"# Sampler: ActionPlan\n")
        f.write(f"# Source: {'custom text' if args.text else f'test set (split={SPLIT}, seed={SEED})'}\n")
        f.write(f"# Use GT Duration: {USE_GT_DURATION}\n\n")
        for i, (p, d) in enumerate(zip(prompts, durations)):
            motion_id = motion_ids[i] if motion_ids else f"sample_{i+1}"
            f.write(f"{i+1}. [{d:.2f}s] {motion_id}: {p}\n")

    sampler = _build_pyramid_sampler(
        run_dir, ckpt_path, device, float(GUIDANCE),
        steps_per_block=getattr(args, "steps_per_block", 2),
    )

    # Generate samples and collect metadata for JSON output
    generations: List[Dict[str, Any]] = []
    for idx, (prompt, duration) in enumerate(zip(prompts, durations)):
        safe = _sanitize_filename(prompt)
        prompt_root = os.path.join(out_root, safe)
        os.makedirs(prompt_root, exist_ok=True)

        # Save prompt
        try:
            with open(os.path.join(prompt_root, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(f"Duration: {duration:.2f}s\n")
                f.write(prompt)
        except Exception:
            pass

        logger.info("[%d/%d] Generating (%.2fs): %s", idx + 1, len(prompts), duration, prompt)

        motion_id = motion_ids[idx] if motion_ids and idx < len(motion_ids) else None

        # Generate with our model
        logger.info("  Generating with our model...")
        
        result = sampler.sample(
            text=prompt,
            seconds=float(duration),
            output_dir=None,
            fps=LATENT_FPS,
        )
        if result is None:
            logger.error("  Failed to generate motion for %s", prompt)
            continue

        latents = result.get("features")
        if latents is None:
            logger.error(
                "Expected 'features' in sampler output, got keys: %s",
                list(result.keys()),
            )
            continue

        outputs = latent_pipeline.decode_and_render(
            latents,
            Path(prompt_root),
            safe,
            prompt,
            render=RENDER_MP4,
        )

        if RENDER_MP4:
            logger.info("  Video: %s", outputs["video_path"])
        logger.info("  Decoded motion: %s", outputs["decoded_path"])
        logger.info("  Raw latents: %s", outputs["latents_path"])

        gen_motion_id = motion_id if motion_id else f"sample_{idx+1}"
        generations.append({
            "index": idx + 1,
            "motion_id": gen_motion_id,
            "prompt": prompt,
            "duration": duration,
            "output_dir": prompt_root,
            "decoded_path": outputs.get("decoded_path"),
            "latents_path": outputs.get("latents_path"),
        })

    # Write comprehensive JSON with motion_id and prompt for all generated motions
    generations_path = os.path.join(out_root, "generations.json")
    with open(generations_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sampler": "ActionPlan",
                "num_generated": len(generations),
                "config": {
                    "split": SPLIT,
                    "seed": SEED,
                    "use_gt_duration": USE_GT_DURATION,
                    "guidance": GUIDANCE,
                },
                "generations": generations,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Generations manifest: %s", generations_path)

    logger.info("=" * 60)
    logger.info("Generation complete! Output saved to: %s", out_root)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
