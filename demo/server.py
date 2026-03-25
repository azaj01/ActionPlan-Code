"""
FastAPI server for streaming text-to-motion generation.

Provides SSE endpoints for real-time motion generation with SMPL mesh data.
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import time
import orjson
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sampler = None


def get_default_run_dir() -> str:
    actionplan_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        actionplan_root,
        "outputs",
        "actionplan"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sampler
    logger.info("Initializing ActionPlan streaming block sampler...")

    from streaming_sampler import StreamingARBlockSampler

    run_dir = os.environ.get("ACTIONPLAN_RUN_DIR", get_default_run_dir())
    ckpt_path = os.environ.get("ACTIONPLAN_CKPT_PATH", None)

    sampler = StreamingARBlockSampler(
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        device=os.environ.get("ACTIONPLAN_DEVICE", None),
        guidance_weight=float(os.environ.get("ACTIONPLAN_GUIDANCE", "3.0")),
        overlap_frames=int(os.environ.get("ACTIONPLAN_OVERLAP_FRAMES", "8")),
        steps_per_block=int(os.environ.get("ACTIONPLAN_STEPS_PER_BLOCK", "2")),
        sampling_timesteps=int(os.environ.get("ACTIONPLAN_SAMPLING_TIMESTEPS", "10")),
    )
    logger.info("Sampler initialized successfully")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="ACTIONPLAN Streaming Motion API",
    description="Real-time text-to-motion generation with SMPL mesh streaming",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    text: str
    seconds: float = 5.0
    session_id: Optional[str] = None
    num_blocks: Optional[int] = None


class SessionResponse(BaseModel):
    session_id: str


# Curated HumanML3D-style prompts (used when train annotations are not available)
SAMPLE_PROMPTS_FALLBACK = [
    "a person walks forward slowly",
    "someone waves their hand to say goodbye",
    "a person sits down on a chair",
    "a person stands up from sitting",
    "someone runs forward at a fast pace",
    "a person jumps up and down",
    "someone bends over to pick something up",
    "a person walks backward carefully",
    "someone claps their hands together",
    "a person stretches their arms above their head",
    "someone walks in a circle",
    "a person kicks a ball with their right foot",
    "someone nods their head yes",
    "a person shakes their head no",
    "someone walks and then stops",
    "a person walks forward and waves",
    "someone crouches down low",
    "a person walks sideways to the left",
    "someone punches forward with their right hand",
    "a person walks with their hands in pockets",
    "someone turns around 180 degrees",
    "a person walks up stairs",
    "someone does a jumping jack",
    "a person walks and sits on the ground",
    "someone reaches up to grab something",
    "a person walks slowly with a limp",
    "someone walks forward and then turns left",
    "a person raises both arms to the sides",
    "someone walks in place",
    "a person walks forward and bends down",
    "someone walks and then runs",
    "a person walks backward and waves",
    "someone walks forward with arms swinging",
    "a person stands still and looks around",
    "someone walks in a zigzag pattern",
    "a person walks forward and stops to look",
    "someone walks and then crouches",
    "a person walks with hands on hips",
    "someone walks forward and turns right",
    "a person walks and gestures with hands",
    "someone walks slowly looking down",
    "a person walks and shrugs shoulders",
    "someone walks forward and points",
    "a person walks and scratches head",
    "someone walks and crosses arms",
    "a person walks and rubs hands together",
    "someone walks forward and yawns",
    "a person walks and touches their face",
    "someone walks and looks over shoulder",
    "a person walks and adjusts their shirt",
    "someone walks forward and stops abruptly",
]


def _load_sample_prompts() -> list[str]:
    """Load 50 prompts from train set, or fallback to curated list."""
    actionplan_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    annot_path = os.path.join(actionplan_root, "datasets", "annotations", "humanml3d_latents")
    split_path = os.path.join(annot_path, "splits", "train.txt")
    annot_file = os.path.join(annot_path, "annotations.json")

    if os.path.exists(split_path) and os.path.exists(annot_file):
        try:
            with open(split_path, "r", encoding="utf-8") as f:
                keyids = [line.strip() for line in f if line.strip()]
            with open(annot_file, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            keyids = [k for k in keyids if k in annotations]
            if not keyids:
                logger.warning("No train prompts found, using fallback")
                return SAMPLE_PROMPTS_FALLBACK

            # Sample one prompt per motion ID (different motion IDs, diverse prompts)
            seen_texts: set[str] = set()
            prompts: list[str] = []
            rng = __import__("random").Random(42)
            shuffled = list(keyids)
            rng.shuffle(shuffled)

            for keyid in shuffled:
                if len(prompts) >= 100:
                    break
                anns = annotations.get(keyid, {}).get("annotations", [])
                if not anns:
                    continue
                # Pick one annotation per motion ID (randomly) for diverse motion IDs
                ann = rng.choice(anns)
                text = str(ann.get("text", "")).strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    prompts.append(text)

            if prompts:
                return prompts
        except Exception as e:
            logger.warning("Failed to load train prompts: %s, using fallback", e)

    return SAMPLE_PROMPTS_FALLBACK


_sample_prompts_cache: list[str] | None = None


def get_sample_prompts() -> list[str]:
    global _sample_prompts_cache
    if _sample_prompts_cache is None:
        _sample_prompts_cache = _load_sample_prompts()
    return _sample_prompts_cache


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "sampler_loaded": sampler is not None},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/sample-prompts")
async def get_sample_prompts_endpoint():
    """Return 150 sample prompts from the train set (or curated fallback)."""
    return JSONResponse(
        content={"prompts": get_sample_prompts()},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/session/create")
async def create_session() -> SessionResponse:
    """Create a new generation session."""
    session_id = str(uuid.uuid4())
    if sampler:
        sampler.get_or_create_session(session_id)
    return SessionResponse(session_id=session_id)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clear a session's accumulated state."""
    if sampler:
        sampler.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


class TrimRequest(BaseModel):
    keep_frames: int


@app.post("/session/{session_id}/trim")
async def trim_session(session_id: str, body: TrimRequest):
    """Trim session to keep only the first keep_frames decoded frames. Removes the latest generation(s)."""
    if sampler is None:
        raise HTTPException(status_code=503, detail="Sampler not initialized")
    ok, msg = sampler.trim_session(session_id, body.keep_frames)
    if not ok:
        logger.warning("Trim failed: session_id=%s keep_frames=%s msg=%s", session_id, body.keep_frames, msg)
        if msg == "session_not_found":
            raise HTTPException(status_code=404, detail="Session not found")
        raise HTTPException(status_code=400, detail="Trim failed")
    return {"status": msg, "session_id": session_id, "keep_frames": body.keep_frames}


@app.get("/mesh/faces")
async def get_mesh_faces():
    """Get SMPL mesh faces for Three.js rendering."""
    if sampler is None:
        raise HTTPException(status_code=503, detail="Sampler not initialized")
    
    faces = sampler.get_smpl_faces()
    return JSONResponse(
        content={"faces": faces.tolist()},
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/session/{session_id}/vertices")
async def get_session_vertices(session_id: str):
    """Get all accumulated vertices for a session."""
    if sampler is None:
        raise HTTPException(status_code=503, detail="Sampler not initialized")
    
    vertices = sampler.get_session_vertices(session_id)
    if vertices is None:
        raise HTTPException(status_code=404, detail="Session not found or no vertices")
    
    return JSONResponse(content={
        "vertices": vertices.tolist(),
        "num_frames": vertices.shape[0],
        "fps": sampler.DECODED_FPS,
    })


async def generate_sse_stream(request: GenerateRequest):
    """Generate SSE events for streaming motion generation."""
    if sampler is None:
        yield b"data: " + orjson.dumps({'type': 'error', 'message': 'Sampler not initialized'}) + b"\n\n"
        return
    
    try:
        t_last = time.perf_counter()
        for block_data in sampler.sample_streaming(
            text=request.text,
            seconds=request.seconds,
            session_id=request.session_id,
            num_blocks=request.num_blocks,
        ):
            t_send_start = time.perf_counter()
            elapsed_since_prev = t_send_start - t_last
            payload = b"data: " + orjson.dumps(block_data) + b"\n\n"
            yield payload
            t_send_end = time.perf_counter()
            send_duration = t_send_end - t_send_start
            t_last = t_send_end

            if block_data.get("type") == "block_complete":
                block_idx = block_data.get("block_idx", "?")
                total = block_data.get("total_blocks", "?")
                payload_size = len(payload) if isinstance(payload, bytes) else len(payload.encode())
                logger.info(
                    f"Block {block_idx + 1}/{total} sent | "
                    f"since prev: {elapsed_since_prev:.3f}s | "
                    f"serialize+send: {send_duration:.3f}s | "
                    f"payload: {payload_size / 1e6:.2f} MB"
                )
            await asyncio.sleep(0)
            
    except Exception as e:
        logger.exception("Error during generation")
        yield b"data: " + orjson.dumps({'type': 'error', 'message': str(e)}) + b"\n\n"


def _pack_ws_binary_event(event: dict, vertices_bin: bytes) -> bytes:
    """
    Pack a binary WS message as:
      [header_len:uint32 little-endian][header_json:utf8 bytes][vertex_payload:bytes]
    """
    header_bytes = orjson.dumps(event)
    header_len = len(header_bytes).to_bytes(4, "little")
    return header_len + header_bytes + vertices_bin


async def _await_ws_ack(websocket: WebSocket, expected_block_idx: int) -> float:
    """
    Wait until client ACKs a specific block index.
    Returns wait duration in seconds.
    """
    t0 = time.perf_counter()
    while True:
        raw = await websocket.receive_text()
        try:
            msg = orjson.loads(raw)
        except Exception:
            continue
        if msg.get("type") == "ack" and msg.get("block_idx") == expected_block_idx:
            return time.perf_counter() - t0


@app.websocket("/ws/generate")
async def generate_ws(websocket: WebSocket):
    await websocket.accept()

    if sampler is None:
        await websocket.send_text(orjson.dumps({"type": "error", "message": "Sampler not initialized"}).decode("utf-8"))
        await websocket.close(code=1011)
        return

    try:
        init_payload = await websocket.receive_text()
        request = GenerateRequest.model_validate_json(init_payload)

        t_last = time.perf_counter()
        for block_data in sampler.sample_streaming(
            text=request.text,
            seconds=request.seconds,
            session_id=request.session_id,
            num_blocks=request.num_blocks,
            vertices_format="binary",
            include_final_vertices=False,
        ):
            t_send_start = time.perf_counter()
            elapsed_since_prev = t_send_start - t_last

            event_type = block_data.get("type")
            if event_type == "block_complete":
                vertices_bin = block_data.pop("vertices_bin", b"")
                payload = _pack_ws_binary_event(block_data, vertices_bin)
                await websocket.send_bytes(payload)
            else:
                payload = orjson.dumps(block_data)
                await websocket.send_text(payload.decode("utf-8"))

            t_send_end = time.perf_counter()
            send_duration = t_send_end - t_send_start
            t_last = t_send_end

            if event_type == "block_complete":
                block_idx = block_data.get("block_idx", "?")
                total = block_data.get("total_blocks", "?")
                ack_wait = await _await_ws_ack(websocket, block_idx)
                logger.info(
                    f"WS Block {block_idx + 1}/{total} sent | "
                    f"since prev: {elapsed_since_prev:.3f}s | "
                    f"send: {send_duration:.3f}s | "
                    f"ack_wait: {ack_wait:.3f}s | "
                    f"payload: {len(payload) / 1e6:.2f} MB"
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("Error during WebSocket generation")
        try:
            await websocket.send_text(orjson.dumps({"type": "error", "message": str(e)}).decode("utf-8"))
        except Exception:
            pass
        try:
            await websocket.close(code=1011)
        except Exception:
            pass


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate motion from text with SSE streaming.
    
    Returns Server-Sent Events with block data as generation progresses.
    """
    return StreamingResponse(
        generate_sse_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/generate/full")
async def generate_full(request: GenerateRequest):
    """
    Generate complete motion and return all data at once.
    
    Useful for non-streaming clients.
    """
    if sampler is None:
        raise HTTPException(status_code=503, detail="Sampler not initialized")
    
    blocks = []
    metadata = {}
    
    for block_data in sampler.sample_streaming(
        text=request.text,
        seconds=request.seconds,
        session_id=request.session_id,
        num_blocks=request.num_blocks,
    ):
        if block_data["type"] == "generation_start":
            metadata["start"] = block_data
        elif block_data["type"] == "block_complete":
            blocks.append(block_data)
        elif block_data["type"] == "generation_complete":
            metadata["complete"] = block_data
    
    return JSONResponse(content={
        "metadata": metadata,
        "blocks": blocks,
    })


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
