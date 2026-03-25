"""
G1 / Sonic ZMQ streaming adapter for ActionPlan ``generate.py --streaming --g1``.

Publishes body pose as Sonic ZMQ protocol v3 (same layout as ``zmq_post/streaming_smpl_publisher.py``).
Uses ``deps/smplh/SMPLH_NEUTRAL.npz`` via the ``smplx`` PyTorch body-model package (SMPL-H, not SMPL-X).
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import smplx
import torch
import zmq
from scipy.spatial.transform import Rotation as R

from src.tools.sonic_g1_zmq.pico_utils import compute_from_body_poses, _pack_pose_v3

_NUM_ROBOT_JOINTS = 29
_FRAME_WINDOW_SIZE = 10


def g1_smplh_model_path() -> Path:
    """Fixed path: ``<repo_root>/deps/smplh/SMPLH_NEUTRAL.npz`` (same assets as ``SMPLH`` elsewhere)."""
    return Path(__file__).resolve().parents[2] / "deps" / "smplh" / "SMPLH_NEUTRAL.npz"


def resolve_g1_body_model_path() -> str:
    p = g1_smplh_model_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"SMPL-H neutral model not found: {p}. Add SMPLH_NEUTRAL.npz under deps/smplh/."
        )
    return str(p)


# --- Quaternion interpolation (numpy, matches streaming_smpl_publisher) ---


def _aa_to_quat(aa):
    angles = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = angles * 0.5
    small = angles < 1e-6
    scale = np.where(
        small,
        0.5 - angles**2 / 48.0,
        np.sin(half) / np.where(small, 1.0, angles),
    )
    return np.concatenate([np.cos(half), aa * scale], axis=-1)


def _quat_to_aa(q):
    w = np.clip(q[..., :1], -1.0, 1.0)
    xyz = q[..., 1:]
    half = np.arccos(w)
    sin_half = np.sin(half)
    small = np.abs(sin_half) < 1e-6
    axis = np.where(small, xyz, xyz / np.where(small, 1.0, sin_half))
    return axis * (2.0 * half)


def _slerp_quat(q0, q1, t=0.5):
    cos_h = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(cos_h < 0, -q1, q1)
    cos_h = np.abs(np.clip(cos_h, -1.0, 1.0))
    half = np.arccos(cos_h)
    sin_h = np.sin(half)
    near = np.abs(sin_h) < 1e-4
    ra = np.where(near, 1.0 - t, np.sin((1.0 - t) * half) / np.where(near, 1.0, sin_h))
    rb = np.where(near, t, np.sin(t * half) / np.where(near, 1.0, sin_h))
    return ra * q0 + rb * q1


def interp_smpl_frame(poses0, trans0, poses1, trans1, t=0.5):
    aa0 = poses0.reshape(22, 3)
    aa1 = poses1.reshape(22, 3)
    q0 = _aa_to_quat(aa0)
    q1 = _aa_to_quat(aa1)
    qi = _slerp_quat(q0, q1, t)
    aa_i = _quat_to_aa(qi)
    trans_i = (1.0 - t) * trans0 + t * trans1
    return aa_i.reshape(66).astype(np.float32), trans_i.astype(np.float32)


def _extract_yaw(rotvec):
    mat = R.from_rotvec(rotvec).as_matrix()
    fwd = mat[:, 2]
    return float(np.arctan2(fwd[0], fwd[2]))


def smpl_frame_to_zmq(poses_t, trans_t, frame_idx, smpl_model):
    """One SMPL frame -> Sonic v3 field dict + debug arrays (matches zmq reference)."""
    global_orient_np = poses_t[:3].astype(np.float64)
    body_pose_np = poses_t[3:66].astype(np.float32)

    actionplan2amass = np.array([[-1, 0.0, 0], [0, 0, 1], [0, 1, 0]])
    amass2yup = np.array([[-1, 0, 0.0], [0, 0, 1], [0, 1, 0]])
    mat_comb = np.matmul(amass2yup, actionplan2amass)
    global_orient_np = R.from_matrix(
        np.matmul(mat_comb, R.from_rotvec(global_orient_np).as_matrix())
    ).as_rotvec()
    trans_t = np.matmul(mat_comb, trans_t.reshape((3, -1))).reshape((3,))

    # 3 + 63 from streamer272_to_smpl matches SMPL-H body_pose (21 joints); hands fixed at zero.
    # Quat stack for compute_from_body_poses pads to 24×3 like the original SMPL path.
    body_pose_padded = np.concatenate([body_pose_np, np.zeros(6, dtype=np.float32)])
    with torch.no_grad():
        smpl_output = smpl_model(
            betas=torch.zeros(1, 10),
            body_pose=torch.from_numpy(body_pose_np).unsqueeze(0),
            global_orient=torch.from_numpy(global_orient_np.astype(np.float32)).unsqueeze(0),
            transl=torch.from_numpy(trans_t.astype(np.float32)).unsqueeze(0),
            left_hand_pose=torch.zeros(1, 45),
            right_hand_pose=torch.zeros(1, 45),
            return_full_pose=True,
        )
    joint_pos = np.zeros((1, _NUM_ROBOT_JOINTS), dtype=np.float32)
    joint_vel = np.zeros((1, _NUM_ROBOT_JOINTS), dtype=np.float32)

    body_joints = smpl_output.joints[:, :24].cpu().numpy()
    quat = (
        R.from_rotvec(np.concatenate([global_orient_np, body_pose_padded]).reshape((-1, 3)))
        .as_quat()
        .reshape((-1, 24, 4))
    )
    data_dict = compute_from_body_poses([], "cpu", np.concatenate([body_joints, quat], -1)[0])
    joint_pos[:, 23:] = data_dict["smpl_joints_local"][:, -2:].reshape(-1, 6).cpu().numpy()

    frame_data = {
        "body_quat": data_dict["global_orient_quat"].cpu().numpy(),
        "frame_index": np.array([frame_idx], dtype=np.int32),
        "smpl_joints": data_dict["smpl_joints_local"].cpu().numpy(),
        "smpl_pose": data_dict["smpl_pose"][:, :63].reshape(-1, 21, 3).cpu().numpy(),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }
    vertices = smpl_output.vertices[0].cpu().numpy()
    joints_local = data_dict["smpl_joints_local"][0].cpu().numpy()
    body_quat = data_dict["global_orient_quat"][0].cpu().numpy()
    return frame_data, vertices, joints_local, body_quat


def pack_frame_window(frame_buffer: List[dict], topic: str = "pose") -> bytes:
    _KEYS = ("body_quat", "frame_index", "smpl_joints", "smpl_pose", "joint_pos", "joint_vel")
    stacked = {k: np.concatenate([f[k] for f in frame_buffer], axis=0) for k in _KEYS}
    return _pack_pose_v3(
        body_quat=stacked["body_quat"],
        frame_index=stacked["frame_index"],
        smpl_joints=stacked["smpl_joints"],
        smpl_pose=stacked["smpl_pose"],
        joint_pos=stacked["joint_pos"],
        joint_vel=stacked["joint_vel"],
        topic=topic,
    )


@dataclass
class G1ZmqFrameState:
    """Carries continuity between streaming batches and text prompts."""

    prev_T: int = 0
    gen_frame_idx: int = 0
    last_world_poses: Optional[np.ndarray] = None
    last_world_trans: Optional[np.ndarray] = None
    interp_prev_poses: Optional[np.ndarray] = None
    interp_prev_trans: Optional[np.ndarray] = None
    world_R_delta: Optional[R] = None
    world_R_mat: Optional[np.ndarray] = None
    world_xz_offset: Optional[np.ndarray] = None

    def reset_for_new_prompt(self, cond_frames: int) -> None:
        self.prev_T = cond_frames
        self.world_R_delta = None
        self.world_R_mat = None
        self.world_xz_offset = None
        self.interp_prev_poses = None
        self.interp_prev_trans = None

    def finalize_prompt(self) -> None:
        if self.interp_prev_poses is not None:
            self.last_world_poses = self.interp_prev_poses.copy()
            self.last_world_trans = self.interp_prev_trans.copy()

    def full_reset(self) -> None:
        self.prev_T = 0
        self.gen_frame_idx = 0
        self.last_world_poses = None
        self.last_world_trans = None
        self.interp_prev_poses = None
        self.interp_prev_trans = None
        self.world_R_delta = None
        self.world_R_mat = None
        self.world_xz_offset = None


def expand_poses_for_zmq_queue(
    poses_np: np.ndarray,
    trans_np: np.ndarray,
    state: G1ZmqFrameState,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Take cumulative SMPL sequences from ``streamer272_to_smpl``, emit new frames only
    (with world stitching + interpolation) for the ZMQ publisher queue.
    """
    prev_T = state.prev_T
    new_T = len(poses_np)
    new_poses = poses_np[prev_T:new_T].copy()
    new_trans = trans_np[prev_T:new_T].copy()
    M = len(new_poses)
    out: List[Tuple[np.ndarray, np.ndarray, int]] = []

    if M == 0:
        return out

    if state.last_world_poses is not None and state.world_R_delta is None:
        yaw_last = _extract_yaw(state.last_world_poses[:3])
        yaw_first = _extract_yaw(new_poses[0, :3])
        state.world_R_delta = R.from_euler("y", yaw_last - yaw_first)
        state.world_R_mat = state.world_R_delta.as_matrix()
        rotated_first = state.world_R_mat @ new_trans[0]
        state.world_xz_offset = np.array(
            [
                state.last_world_trans[0] - rotated_first[0],
                0.0,
                state.last_world_trans[2] - rotated_first[2],
            ]
        )

    if state.world_R_mat is not None:
        assert state.world_R_delta is not None and state.world_xz_offset is not None
        wrd = state.world_R_delta
        wmat = state.world_R_mat
        off = state.world_xz_offset
        for i in range(M):
            new_poses[i, :3] = (wrd * R.from_rotvec(new_poses[i, :3])).as_rotvec()
            new_trans[i] = wmat @ new_trans[i] + off

    if state.interp_prev_poses is not None:
        mid_p, mid_t = interp_smpl_frame(
            state.interp_prev_poses,
            state.interp_prev_trans,
            new_poses[0],
            new_trans[0],
        )
        out.append((mid_p, mid_t, state.gen_frame_idx))
        state.gen_frame_idx += 1

    out.append((new_poses[0], new_trans[0], state.gen_frame_idx))
    state.gen_frame_idx += 1
    for i in range(M - 1):
        mid_p, mid_t = interp_smpl_frame(
            new_poses[i],
            new_trans[i],
            new_poses[i + 1],
            new_trans[i + 1],
        )
        out.append((mid_p, mid_t, state.gen_frame_idx))
        state.gen_frame_idx += 1
        out.append((new_poses[i + 1], new_trans[i + 1], state.gen_frame_idx))
        state.gen_frame_idx += 1

    state.interp_prev_poses = new_poses[-1]
    state.interp_prev_trans = new_trans[-1]
    state.prev_T = new_T
    return out


@dataclass
class _PublisherConfig:
    host: str
    port: int
    topic: str
    hz: float
    frame_window_size: int = _FRAME_WINDOW_SIZE


class G1ZmqPublisher:
    """
    Background thread: dequeue (poses66, trans3, frame_idx), pack v3 window, PUB send at fixed Hz.
    """

    def __init__(
        self,
        smpl_model: "smplx.SMPL",
        host: str = "*",
        port: int = 5556,
        topic: str = "pose",
        hz: float = 30.0,
        frame_window_size: int = _FRAME_WINDOW_SIZE,
    ) -> None:
        self.smpl_model = smpl_model
        self.cfg = _PublisherConfig(
            host=host, port=port, topic=topic, hz=hz, frame_window_size=frame_window_size
        )
        self._stop = threading.Event()
        self._q: queue.Queue[Tuple[np.ndarray, np.ndarray, int]] = queue.Queue(maxsize=600)
        self._thread: Optional[threading.Thread] = None
        self._sock = None
        self._ctx = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="G1ZmqPublisher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

    def enqueue(self, poses: np.ndarray, trans: np.ndarray, frame_idx: int) -> None:
        try:
            self._q.put((poses, trans, frame_idx), timeout=0.5)
        except queue.Full:
            pass

    def _run(self) -> None:
        ctx = zmq.Context()
        self._ctx = ctx
        sock = ctx.socket(zmq.PUB)
        self._sock = sock
        sock.bind(f"tcp://{self.cfg.host}:{self.cfg.port}")
        print(
            f"[g1-zmq] Bound tcp://{self.cfg.host}:{self.cfg.port} "
            f"topic={self.cfg.topic!r} rate={self.cfg.hz} Hz"
        )
        print("[g1-zmq] Waiting 300 ms for ZMQ subscribers to connect ...")
        time.sleep(0.3)

        interval = 1.0 / max(self.cfg.hz, 1e-6)
        last_poses: Optional[np.ndarray] = None
        last_trans: Optional[np.ndarray] = None
        last_frame_idx = 0
        frame_buffer: Deque[dict] = deque(maxlen=self.cfg.frame_window_size)

        while not self._stop.is_set():
            t0 = time.monotonic()
            got_new = False
            try:
                last_poses, last_trans, last_frame_idx = self._q.get_nowait()
                got_new = True
            except queue.Empty:
                pass

            if last_poses is None:
                time.sleep(interval)
                continue

            try:
                frame_data, _, _, _ = smpl_frame_to_zmq(
                    last_poses, last_trans, last_frame_idx, self.smpl_model
                )
                if got_new:
                    frame_buffer.append(frame_data)
                if len(frame_buffer) == 0:
                    time.sleep(interval)
                    continue
                msg = pack_frame_window(list(frame_buffer), topic=self.cfg.topic)
                sock.send(msg)
            except Exception as e:
                print(f"[g1-zmq] Conversion/send error: {e}")

            elapsed = time.monotonic() - t0
            rem = interval - elapsed
            if rem > 0:
                time.sleep(rem)


def create_g1_publisher(
    host: str,
    port: int,
    topic: str,
    hz: float,
) -> G1ZmqPublisher:
    path = resolve_g1_body_model_path()
    print(f"[g1-zmq] Loading SMPL-H from {path} ...")
    smpl_model = smplx.create(
        path,
        model_type="smplh",
        gender="neutral",
        num_betas=10,
        ext="npz",
        use_pca=False,
    )
    smpl_model.eval()
    pub = G1ZmqPublisher(smpl_model, host=host, port=port, topic=topic, hz=hz)
    pub.start()
    return pub
