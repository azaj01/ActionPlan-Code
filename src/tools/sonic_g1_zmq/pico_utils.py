import torch
import numpy as np 
import json
from scipy.spatial.transform import Rotation as R, Rotation as sRot

from .torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion
)
from .rotations import remove_smpl_base_rot, smpl_root_ytoz_up

_HEADER_SIZE = 1280

def _pack_pose_v3(
    body_quat: np.ndarray,   # [N, 4]       root quaternion (w, x, y, z)
    frame_index: np.ndarray, # [N]           monotonically increasing
    smpl_joints: np.ndarray, # [N, 24, 3]   pelvis-relative joint positions, Z-up
    smpl_pose: np.ndarray,   # [N, 21, 3]   body joint axis-angles
    joint_pos: np.ndarray,   # [N, 29]      robot joint positions (zeros ok)
    joint_vel: np.ndarray,   # [N, 29]      robot joint velocities (zeros ok)
    topic: str = "pose",
) -> bytes:
    """Pack arrays into the ZMQ Protocol v3 binary message format.

    Wire layout:  [topic bytes] [1280-byte JSON header] [packed binary payload]

    The C++ receiver reads the header to know each field's name, dtype, and
    shape, then deserialises the packed payload accordingly.
    """
    arrays = [
        ("body_quat",   body_quat.astype(np.float32),   "f32"),
        ("frame_index", frame_index.astype(np.int32),    "i32"),
        ("joint_pos",   joint_pos.astype(np.float32),    "f32"),
        ("joint_vel",   joint_vel.astype(np.float32),    "f32"),
        ("smpl_joints", smpl_joints.astype(np.float32),  "f32"),
        ("smpl_pose",   smpl_pose.astype(np.float32),    "f32"),
    ]
    fields, payloads = [], []
    for name, arr, dtype in arrays:
        arr = np.ascontiguousarray(arr)
        fields.append({"name": name, "dtype": dtype, "shape": list(arr.shape)})
        payloads.append(arr.tobytes())

    header_json = json.dumps(
        {"v": 3, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(header_json) <= _HEADER_SIZE, f"Header too large ({len(header_json)} bytes)"
    header_bytes = header_json.ljust(_HEADER_SIZE, b"\x00")

    return topic.encode("utf-8") + header_bytes + b"".join(payloads)


# R_BASE = quaternion [w=0.5, x=0.5, y=0.5, z=0.5] — SMPL rest-pose base rotation.
# Applying conj(R_BASE) right-multiplied to q_zup is `remove_smpl_base_rot`.
_R_BASE = R.from_quat([0.5, 0.5, 0.5, 0.5])   # scipy uses (x, y, z, w)


def _smpl_global_orient_to_root_quat_zup(global_orient: np.ndarray) -> np.ndarray:
    """Convert SMPL global_orient (axis-angle, Y-up) → body_quat (w,x,y,z, Z-up).

    Matches the reference pipeline:
      1. smpl_root_ytoz_up: 90° X rotation (Y-up → Z-up)
      2. remove_smpl_base_rot: multiply by conj(R_BASE)

    Args:
        global_orient: shape [3], axis-angle in radians (SMPL Y-up convention).

    Returns:
        shape [4], quaternion (w, x, y, z) matching what the deployment expects.
    """
    rot_yup  = R.from_rotvec(global_orient.astype(float))
    q_x90    = R.from_euler("x", np.pi / 2)
    rot_zup  = q_x90 * rot_yup                  # smpl_root_ytoz_up
    rot_adj  = rot_zup * _R_BASE.inv()           # remove_smpl_base_rot
    x, y, z, w = rot_adj.as_quat()              # scipy returns (x, y, z, w)
    return np.array([w, x, y, z], dtype=np.float32)

# Rotation matrix: OpenCV (x-right, y-down, z-forward) → SMPL Y-up (x-right, y-up, z-backward)
_R_OPENCV_TO_YUP = np.array([[1,  0,  0],
                               [0, -1,  0],
                               [0,  0, -1]], dtype=np.float32)


def opencv_to_yup_aa(aa_cv: np.ndarray) -> np.ndarray:
    """Convert an axis-angle vector from OpenCV to SMPL Y-up convention.

    Args:
        aa_cv: shape [..., 3], axis-angle in OpenCV frame (x-right, y-down, z-forward).

    Returns:
        shape [..., 3], axis-angle in SMPL Y-up frame (x-right, y-up, z-backward).
    """
    return (aa_cv.reshape(-1, 3) @ _R_OPENCV_TO_YUP.T).reshape(aa_cv.shape)


def opencv_to_yup_points(pts_cv: np.ndarray) -> np.ndarray:
    """Convert 3-D points from OpenCV to SMPL Y-up convention.

    Args:
        pts_cv: shape [..., 3], positions in OpenCV frame.

    Returns:
        shape [..., 3], positions in SMPL Y-up frame.
    """
    return (pts_cv.reshape(-1, 3) @ _R_OPENCV_TO_YUP.T).reshape(pts_cv.shape)


# ---------------------------------------------------------------------------
# Main publisher loop
# ---------------------------------------------------------------------------

def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor, shape (T, 69)
        global_orient: Global orientation tensor, shape (T, 3)
        transl: Translation tensor, shape (T, 3)

    Returns:
        Dictionary with processed joints and parameters
    """
    # Convert global_orient to quaternion and apply transformations (robust if utils missing)
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    # Compute joints and vertices using SMPL model (single forward pass)
    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (*, 24, 3)

    # Apply base rotation removal and compute local joints
    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    # ActionPlan: Global orient after zup: [-95.33961714  -2.72060293 179.98251456] after base removal: [-5.33879336e+00 -1.74651777e-02  9.27206212e+01]
    # AMASS: Global orient after zup: [87.37571994  1.46839328 -0.92257429] after base removal: [ -2.64793187   0.9222714  -88.53142476], final sent out: [ 2.64793197 -0.92227116 91.46858718]
    # print("Global orient after zup:", R.from_rotvec(global_orient_new[0].numpy()).as_euler('XYZ', degrees=True),
    #       'after base removal:', R.from_matrix(quaternion_to_rotation_matrix(global_orient_quat)[0].numpy()).as_euler('XYZ', degrees=True))
    
    # # now apply rotation around z by 180: with this it now looks better! 
    align_transform = torch.from_numpy(sRot.from_euler("z", 180, degrees=True).as_matrix()).float()
    global_orient_quat = rotation_matrix_to_quaternion(torch.matmul(align_transform[None], quaternion_to_rotation_matrix(global_orient_quat)))
    joints = torch.matmul(joints, align_transform.T[None]) # this is also needed to have correct orientation 


    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
        "adjusted_transl": transl,
    }


def compute_from_body_poses(parent_indices: list, device, body_poses_np: np.ndarray):
    """
    Compute local joints and body orientation from provided body_poses_np.
    body_poses_np: loaded from xrt.get_body_joints_pose()

    body_poses_np: (N, 7) for N body joints, already in local poses 

    """

    positions = body_poses_np[:, :3]

    # input already in local rotations (w, x, y, z); scipy expects (x, y, z, w) without keyword args (older scipy).
    pose_quats = body_poses_np[:, [6, 3, 4, 5]]  # wxyz
    pose_quats_xyzw = pose_quats[:, [1, 2, 3, 0]]
    # DO NOT do right multiply, this causes instability!
    pose_aa = sRot.from_quat(pose_quats_xyzw).as_rotvec()
    # pose_aa[0:1] = global_orient.as_rotvec()
    # already in local coordinates 

    # global_quats = body_poses_np[:, [6, 3, 4, 5]] # originally was xyzw, now convert to wxyz, these are local poses 
    # # Convert to local rotations
    # global_rots = sRot.from_quat(global_quats, scalar_first=True)
    # global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    # local_rots = []
    # for i in range(24):
    #     if parent_indices[i] == -1:
    #         local_rots.append(global_rots[i])
    #     else:
    #         local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
    #         local_rots.append(local_rot)

    # pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(positions[0]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)
