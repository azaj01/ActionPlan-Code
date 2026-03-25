"""
SMPL mesh utilities for web streaming.

Provides functions to export and compress mesh data for efficient transfer.
"""

import os
import json
import base64
import struct
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np


def get_smpl_faces_path() -> Path:
    """Get path to cached SMPL faces file."""
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "smpl_faces.json"


def cache_smpl_faces(faces: np.ndarray) -> None:
    """Cache SMPL faces to disk for fast loading."""
    faces_path = get_smpl_faces_path()
    with open(faces_path, "w") as f:
        json.dump(faces.astype(int).tolist(), f)


def load_cached_faces() -> Optional[np.ndarray]:
    """Load cached SMPL faces if available."""
    faces_path = get_smpl_faces_path()
    if faces_path.exists():
        with open(faces_path, "r") as f:
            return np.array(json.load(f), dtype=np.int32)
    return None


def compress_vertices_binary(vertices: np.ndarray) -> bytes:
    """
    Compress vertices to binary format for efficient transfer.
    
    Format: [num_frames(4), num_vertices(4), data(float32)]
    """
    num_frames, num_vertices, _ = vertices.shape
    header = struct.pack("<II", num_frames, num_vertices)
    data = vertices.astype(np.float32).tobytes()
    return header + data


def decompress_vertices_binary(data: bytes) -> np.ndarray:
    """Decompress binary vertex data."""
    num_frames, num_vertices = struct.unpack("<II", data[:8])
    vertices_flat = np.frombuffer(data[8:], dtype=np.float32)
    return vertices_flat.reshape(num_frames, num_vertices, 3)


def vertices_to_base64(vertices: np.ndarray) -> str:
    """Encode vertices as base64 string for JSON transport."""
    binary = compress_vertices_binary(vertices)
    return base64.b64encode(binary).decode("ascii")


def base64_to_vertices(b64_str: str) -> np.ndarray:
    """Decode base64 vertices string."""
    binary = base64.b64decode(b64_str)
    return decompress_vertices_binary(binary)


def compute_vertex_normals(
    vertices: np.ndarray, 
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute per-vertex normals for smooth shading.
    
    Args:
        vertices: [N, 3] vertex positions
        faces: [F, 3] face indices
        
    Returns:
        normals: [N, 3] normalized vertex normals
    """
    normals = np.zeros_like(vertices)
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    
    return normals


def compute_bounding_box(vertices: np.ndarray) -> Dict[str, Any]:
    """Compute bounding box for camera positioning."""
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    
    return {
        "min": min_coords.tolist(),
        "max": max_coords.tolist(),
        "center": center.tolist(),
        "size": size.tolist(),
        "diagonal": float(np.linalg.norm(size)),
    }


def prepare_mesh_for_frontend(
    vertices: np.ndarray,
    faces: np.ndarray,
    compute_normals: bool = True,
) -> Dict[str, Any]:
    """
    Prepare mesh data for frontend consumption.
    
    Returns a dictionary with vertices, faces, and optional normals.
    """
    result = {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "num_vertices": vertices.shape[0],
        "num_faces": faces.shape[0],
    }
    
    if compute_normals:
        if vertices.ndim == 3:
            normals = compute_vertex_normals(vertices[0], faces)
        else:
            normals = compute_vertex_normals(vertices, faces)
        result["normals"] = normals.tolist()
    
    return result


def downsample_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample mesh for faster rendering (optional).
    
    Uses simple face decimation - for production use a proper
    mesh decimation library like PyMeshLab.
    """
    if faces.shape[0] <= target_faces:
        return vertices, faces
    
    step = max(1, faces.shape[0] // target_faces)
    decimated_faces = faces[::step]
    
    used_vertices = np.unique(decimated_faces.flatten())
    vertex_map = np.zeros(vertices.shape[0], dtype=np.int32)
    vertex_map[used_vertices] = np.arange(len(used_vertices))
    
    new_vertices = vertices[used_vertices]
    new_faces = vertex_map[decimated_faces]
    
    return new_vertices, new_faces


class MeshStreamEncoder:
    """Encodes mesh data for efficient SSE streaming."""
    
    def __init__(self, faces: np.ndarray):
        self.faces = faces.astype(np.int32)
        self._faces_sent = False
    
    def encode_init_message(self) -> Dict[str, Any]:
        """Encode initial message with mesh topology."""
        self._faces_sent = True
        return {
            "type": "mesh_init",
            "faces": self.faces.tolist(),
            "num_faces": int(self.faces.shape[0]),
        }
    
    def encode_frame(
        self,
        vertices: np.ndarray,
        frame_idx: int,
        include_normals: bool = False,
    ) -> Dict[str, Any]:
        """Encode a single frame's vertex data."""
        msg = {
            "type": "frame",
            "frame_idx": frame_idx,
            "vertices": vertices.tolist(),
        }
        
        if include_normals:
            normals = compute_vertex_normals(vertices, self.faces)
            msg["normals"] = normals.tolist()
        
        return msg
    
    def encode_block(
        self,
        vertices: np.ndarray,
        block_idx: int,
        start_frame: int,
    ) -> Dict[str, Any]:
        """Encode a block of frames."""
        return {
            "type": "block",
            "block_idx": block_idx,
            "start_frame": start_frame,
            "num_frames": vertices.shape[0],
            "vertices": vertices.tolist(),
        }
