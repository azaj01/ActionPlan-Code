"""
272-dimensional Motion Representation utilities for inference and visualization.

This module provides runtime utilities for working with the 272-dim motion
representation from MotionStreamer.

Reference: https://github.com/Li-xingXiao/272-dim-Motion-Representation

The 272-dim representation structure (from recover_visualize.py):
- [0:2]         - local XZ velocities of root (no heading)
- [2:8]         - heading as 6D rotation (differential, accumulated to get full heading)
- [8:74]        - local joint positions (22 joints × 3D = 66 dims), no heading, at XZ origin
- [74:140]      - local joint velocities (22 joints × 3D = 66 dims)
- [140:272]     - local joint rotations in 6D (22 joints × 6D = 132 dims), no heading

Total: 2 + 6 + 66 + 66 + 132 = 272 dimensions

Coordinate system: Y-up (Y is vertical, XZ is ground plane)
"""

import torch
import numpy as np
import einops
from torch import Tensor
from typing import Dict, Tuple, Optional

from .geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)

# Feature indices (matching recover_visualize.py from original repo)
ROOT_VEL_XZ_IDX = slice(0, 2)      # Local XZ velocities (2 dims)
HEADING_6D_IDX = slice(2, 8)       # Heading as 6D rotation (6 dims)
JOINT_POS_IDX = slice(8, 74)       # Local joint positions (22*3=66 dims)
JOINT_VEL_IDX = slice(74, 140)     # Local joint velocities (22*3=66 dims)
JOINT_ROT_IDX = slice(140, 272)    # Local joint rotations 6D (22*6=132 dims)

# Number of joints
NUM_JOINTS = 22


def ungroup_streamer272(features: Tensor) -> Tuple[Tensor, ...]:
    """
    Decompose 272-dim features into components.
    
    Based on recover_visualize.py from:
    https://github.com/Li-xingXiao/272-dim-Motion-Representation
    
    Args:
        features: [T, 272] or [B, T, 272] motion features
        
    Returns:
        Tuple of:
            - vel_xz: [T, 2] or [B, T, 2] - local XZ velocities
            - heading_6d: [T, 6] or [B, T, 6] - heading as 6D rotation (differential)
            - joints_local: [T, 22, 3] or [B, T, 22, 3] - local joint positions
            - joints_vel: [T, 22, 3] or [B, T, 22, 3] - local joint velocities
            - poses_6d: [T, 22, 6] or [B, T, 22, 6] - local joint rotations 6D
    """
    batched = features.dim() == 3
    
    if batched:
        B, T, D = features.shape
        assert D == 272, f"Expected 272 features, got {D}"
        
        vel_xz = features[:, :, 0:2]  # [B, T, 2] - XZ velocities
        heading_6d = features[:, :, 2:8]  # [B, T, 6] - heading 6D rotation
        joints_local_flat = features[:, :, 8:74]  # [B, T, 66]
        joints_vel_flat = features[:, :, 74:140]  # [B, T, 66]
        poses_6d_flat = features[:, :, 140:272]  # [B, T, 132]
        
        joints_local = einops.rearrange(joints_local_flat, "b t (j c) -> b t j c", c=3)
        joints_vel = einops.rearrange(joints_vel_flat, "b t (j c) -> b t j c", c=3)
        poses_6d = einops.rearrange(poses_6d_flat, "b t (j c) -> b t j c", c=6)
    else:
        T, D = features.shape
        assert D == 272, f"Expected 272 features, got {D}"
        
        vel_xz = features[:, 0:2]  # [T, 2] - XZ velocities
        heading_6d = features[:, 2:8]  # [T, 6] - heading 6D rotation
        joints_local_flat = features[:, 8:74]  # [T, 66]
        joints_vel_flat = features[:, 74:140]  # [T, 66]
        poses_6d_flat = features[:, 140:272]  # [T, 132]
        
        joints_local = einops.rearrange(joints_local_flat, "t (j c) -> t j c", c=3)
        joints_vel = einops.rearrange(joints_vel_flat, "t (j c) -> t j c", c=3)
        poses_6d = einops.rearrange(poses_6d_flat, "t (j c) -> t j c", c=6)
    
    return (
        vel_xz,
        heading_6d,
        joints_local,
        joints_vel,
        poses_6d,
    )


def group_streamer272(
    vel_xz: Tensor,
    heading_6d: Tensor,
    joints_local: Tensor,
    joints_vel: Tensor,
    poses_6d: Tensor,
) -> Tensor:
    """
    Assemble 272-dim features from components.
    
    Based on the original 272-dim representation structure:
    - [0:2]     - XZ velocities (2)
    - [2:8]     - heading 6D rotation (6)
    - [8:74]    - joint positions (66)
    - [74:140]  - joint velocities (66)
    - [140:272] - joint rotations 6D (132)
    
    Total: 2 + 6 + 66 + 66 + 132 = 272
    """
    batched = joints_local.dim() == 4
    
    if batched:
        joints_local_flat = einops.rearrange(joints_local, "b t j c -> b t (j c)")
        joints_vel_flat = einops.rearrange(joints_vel, "b t j c -> b t (j c)")
        poses_6d_flat = einops.rearrange(poses_6d, "b t j c -> b t (j c)")
        
        features = torch.cat([
            vel_xz,                      # [B, T, 2]
            heading_6d,                  # [B, T, 6]
            joints_local_flat,           # [B, T, 66]
            joints_vel_flat,             # [B, T, 66]
            poses_6d_flat,               # [B, T, 132]
        ], dim=-1)
    else:
        joints_local_flat = einops.rearrange(joints_local, "t j c -> t (j c)")
        joints_vel_flat = einops.rearrange(joints_vel, "t j c -> t (j c)")
        poses_6d_flat = einops.rearrange(poses_6d, "t j c -> t (j c)")
        
        features = torch.cat([
            vel_xz,                      # [T, 2]
            heading_6d,                  # [T, 6]
            joints_local_flat,           # [T, 66]
            joints_vel_flat,             # [T, 66]
            poses_6d_flat,               # [T, 132]
        ], dim=-1)
    
    assert features.shape[-1] == 272
    return features


def accumulate_rotations(relative_rotations: Tensor) -> Tensor:
    """
    Accumulate relative rotations to get overall rotation at each frame.
    
    Based on recover_visualize.py from original repo.
    
    Args:
        relative_rotations: [T, 3, 3] relative rotation matrices
        
    Returns:
        accumulated: [T, 3, 3] accumulated rotation matrices
    """
    T = relative_rotations.shape[0]
    device = relative_rotations.device
    dtype = relative_rotations.dtype
    
    accumulated = [relative_rotations[0]]
    for i in range(1, T):
        # R_total[i] = R_rel[i] @ R_total[i-1]
        accumulated.append(relative_rotations[i] @ accumulated[-1])
    
    return torch.stack(accumulated, dim=0)


def streamer272_to_smpl(features: Tensor) -> Dict[str, Tensor]:
    """
    Convert 272-dim features to SMPL parameters.
    
    Based on recover_from_local_rotation() from recover_visualize.py:
    https://github.com/Li-xingXiao/272-dim-Motion-Representation
    
    Args:
        features: [T, 272] or [B, T, 272] motion features
        
    Returns:
        Dictionary with:
            - poses: [T, 66] or [B, T, 66] axis-angle rotations
            - trans: [T, 3] or [B, T, 3] root translation
            - joints: [T, 22, 3] or [B, T, 22, 3] joint positions
    """
    batched = features.dim() == 3
    
    if batched:
        # Process each sample in batch
        results = []
        for i in range(features.shape[0]):
            result = streamer272_to_smpl(features[i])
            results.append(result)
        return {
            k: torch.stack([r[k] for r in results])
            for k in results[0].keys()
        }
    
    (
        vel_xz,           # [T, 2] - local XZ velocities
        heading_6d,       # [T, 6] - heading as 6D rotation (differential)
        joints_local,     # [T, 22, 3] - local joint positions
        joints_vel,       # [T, 22, 3] - local joint velocities (unused for recovery)
        poses_6d,         # [T, 22, 6] - local joint rotations
    ) = ungroup_streamer272(features)
    
    T = features.shape[0]
    device = features.device
    dtype = features.dtype
    njoint = NUM_JOINTS
    
    # ==================== RECOVER HEADING ====================
    # Convert 6D heading to rotation matrices and accumulate
    heading_rot_diff = rotation_6d_to_matrix(heading_6d)  # [T, 3, 3]
    global_heading_rot = accumulate_rotations(heading_rot_diff)  # [T, 3, 3]
    inv_global_heading_rot = global_heading_rot.transpose(-2, -1)  # [T, 3, 3]
    
    # ==================== RECOVER JOINT POSITIONS ====================
    # Add global heading to local positions
    # positions_with_heading = inv_heading @ positions_local
    positions_with_heading = torch.einsum(
        "tij,tnj->tni",
        inv_global_heading_rot,
        joints_local
    )  # [T, 22, 3]
    
    # ==================== RECOVER ROOT TRANSLATION ====================
    # Convert XZ velocities to XYZ (Y is up, so vel[0]->X, vel[1]->Z)
    vel_xyz = torch.zeros(T, 3, device=device, dtype=dtype)
    vel_xyz[:, 0] = vel_xz[:, 0]  # X velocity
    vel_xyz[:, 2] = vel_xz[:, 1]  # Z velocity (from index 1)
    
    # Apply inverse heading to velocities (except first frame)
    vel_xyz_world = vel_xyz.clone()
    if T > 1:
        vel_xyz_world[1:] = torch.einsum(
            "tij,tj->ti",
            inv_global_heading_rot[:-1],
            vel_xyz[1:]
        )
    
    # Integrate velocities to get translation
    root_translation = torch.cumsum(vel_xyz_world, dim=0)
    
    # Get height from pelvis Y coordinate in local positions
    height = joints_local[:, 0, 1]  # Y coordinate of pelvis
    root_translation[:, 1] = height
    
    # Add root translation to joint positions (only X and Z, Y already has height)
    joints = positions_with_heading.clone()
    joints[:, :, 0] += root_translation[:, 0:1]
    joints[:, :, 2] += root_translation[:, 2:3]
    
    # ==================== RECOVER POSES ====================
    # Convert 6D rotations to matrices
    rotations_matrix = rotation_6d_to_matrix(poses_6d)  # [T, 22, 3, 3]
    
    # Add global heading to root rotation
    rotations_matrix[:, 0] = inv_global_heading_rot @ rotations_matrix[:, 0]
    
    # Convert to axis-angle
    poses_aa = matrix_to_axis_angle(rotations_matrix)  # [T, 22, 3]
    poses = einops.rearrange(poses_aa, "t j c -> t (j c)")  # [T, 66]
    
    # Translation is root_translation
    trans = root_translation
    
    return {
        "poses": poses,
        "trans": trans,
        "joints": joints,
    }


def get_joints_from_streamer272(features: Tensor) -> Tensor:
    """
    Quick extraction of joint positions from 272-dim features.
    
    Args:
        features: [T, 272] or [B, T, 272] motion features
        
    Returns:
        joints: [T, 22, 3] or [B, T, 22, 3] joint positions in world space
    """
    smpl_data = streamer272_to_smpl(features)
    return smpl_data["joints"]


def extract_joint_rotations_6d(features: Tensor) -> Tensor:
    """
    Extract 6D joint rotations directly from features.
    
    Args:
        features: [T, 272] or [B, T, 272] motion features
        
    Returns:
        poses_6d: [T, 22, 6] or [B, T, 22, 6] joint rotations
    """
    _, _, _, _, poses_6d = ungroup_streamer272(features)
    return poses_6d


def extract_local_joints(features: Tensor) -> Tensor:
    """
    Extract local joint positions directly from features.
    
    Args:
        features: [T, 272] or [B, T, 272] motion features
        
    Returns:
        joints_local: [T, 22, 3] or [B, T, 22, 3] local joint positions
    """
    _, _, joints_local, _, _ = ungroup_streamer272(features)
    return joints_local


def extract_foot_contacts(features: Tensor) -> Tensor:
    """
    Extract foot contact labels from features.
    
    NOTE: The original 272-dim representation does NOT include foot contacts.
    This function returns zeros for compatibility.
    
    Args:
        features: [T, 272] or [B, T, 272] motion features
        
    Returns:
        contacts: [T, 4] or [B, T, 4] zero contact labels
    """
    if features.dim() == 3:
        B, T, _ = features.shape
        return torch.zeros(B, T, 4, device=features.device, dtype=features.dtype)
    else:
        T = features.shape[0]
        return torch.zeros(T, 4, device=features.device, dtype=features.dtype)


# =============================================================================
# Utility functions
# =============================================================================


def numpy_to_tensor(arr: np.ndarray) -> Tensor:
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(arr).float()


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()

