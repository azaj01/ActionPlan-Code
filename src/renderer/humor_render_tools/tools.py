# Adapted from STMC: src/renderer/humor_render_tools/tools.py
# Upstream repo: https://github.com/nv-tlabs/stmc
# Source file: https://github.com/nv-tlabs/stmc/blob/main/src/renderer/humor_render_tools/tools.py

import torch
import numpy as np
from tqdm import tqdm
import trimesh
from .parameters import colors, smpl_connections
from .mesh_viewer import MeshViewer

c2c = lambda tensor: tensor.detach().cpu().numpy()  # noqa


def viz_smpl_seq(
    pyrender,
    out_path,
    body,
    #
    start=None,
    end=None,
    #
    imw=720,
    imh=720,
    fps=20,
    use_offscreen=True,
    follow_camera=True,
    progress_bar=tqdm,
    #
    text_overlay=None,
    text_overlay_seq=None,
    text_color=[0, 0, 0],
    text_font_scale=1.0,
    text_thickness=2,
    text_margin=20,
    #
    contacts=None,
    render_body=True,
    render_joints=False,
    render_skeleton=False,
    render_ground=True,
    ground_plane=None,
    wireframe=False,
    RGBA=False,
    joints_seq=None,
    joints_vel=None,
    vtx_list=None,
    points_seq=None,
    points_vel=None,
    static_meshes=None,
    camera_intrinsics=None,
    img_seq=None,
    point_rad=0.015,
    skel_connections=smpl_connections,
    img_extn="png",
    ground_alpha=1.0,
    body_alpha=None,
    mask_seq=None,
    cam_offset=[0.0, 3.5, 1.5],  # [0.0, 4.0, 1.25],
    ground_color0=[0.8, 0.9, 0.9],
    ground_color1=[0.6, 0.7, 0.7],
    skel_color=[0.5, 0.5, 0.5],  # [0.0, 0.0, 1.0],
    joint_rad=0.015,
    point_color=[0.0, 0.0, 1.0],
    joint_color=[0.0, 1.0, 0.0],
    contact_color=[1.0, 0.0, 0.0],
    vertex_color=colors["vertex"],
    # Optional per-frame vertex colors (list/array of RGB or RGBA in [0,1])
    frame_vertex_colors=None,
    # Optional legend entries: list of {"color": [r,g,b], "label": str}
    legend_entries=None,
    render_bodies_static=None,
    render_points_static=None,
    cam_rot=None,
):
    """
    Visualizes the body model output of a smpl sequence.
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    """

    if contacts is not None and torch.is_tensor(contacts):
        contacts = c2c(contacts)

    if render_body or vtx_list is not None:
        nv = body.v.size(1)
        faces = c2c(body.f)
        T = body.v.size(0)
        use_per_frame_colors = frame_vertex_colors is not None
        if use_per_frame_colors:
            fvc = np.asarray(frame_vertex_colors)
            if fvc.shape[0] != T:
                # Fallback to uniform color if length mismatch
                use_per_frame_colors = False
        body_mesh_seq = []
        for i in range(T):
            if use_per_frame_colors:
                cur_color = fvc[i]
            else:
                cur_color = np.asarray(vertex_color)
            cur_colors = np.tile(cur_color, (nv, 1))
            if body_alpha is not None:
                vtx_alpha = np.ones((cur_colors.shape[0], 1)) * body_alpha
                cur_colors = np.concatenate([cur_colors, vtx_alpha], axis=1)
            mesh_i = trimesh.Trimesh(
                vertices=c2c(body.v[i]),
                faces=faces,
                vertex_colors=cur_colors,
                process=False,
            )
            body_mesh_seq.append(mesh_i)

    if render_joints and joints_seq is None:
        # only body joints
        joints_seq = [c2c(body.Jtr[i, :22]) for i in range(body.Jtr.size(0))]
    elif render_joints and torch.is_tensor(joints_seq[0]):
        joints_seq = [c2c(joint_frame) for joint_frame in joints_seq]

    if joints_vel is not None and torch.is_tensor(joints_vel[0]):
        joints_vel = [c2c(joint_frame) for joint_frame in joints_vel]
    if points_vel is not None and torch.is_tensor(points_vel[0]):
        points_vel = [c2c(joint_frame) for joint_frame in points_vel]

    mv = MeshViewer(
        pyrender,
        width=imw,
        height=imh,
        use_offscreen=use_offscreen,
        follow_camera=follow_camera,
        camera_intrinsics=camera_intrinsics,
        img_extn=img_extn,
        default_cam_offset=cam_offset,
        default_cam_rot=cam_rot,
    )
    if text_overlay_seq is not None:
        mv.set_text_overlay_seq(text_overlay_seq)
        mv.set_text_overlay(
            text="",
            color=text_color,
            font_scale=text_font_scale,
            thickness=text_thickness,
            margin=text_margin,
        )
    elif text_overlay is not None:
        mv.set_text_overlay(
            text=str(text_overlay),
            color=text_color,
            font_scale=text_font_scale,
            thickness=text_thickness,
            margin=text_margin,
        )
    if legend_entries is not None:
        try:
            mv.set_legend(legend_entries)
        except Exception:
            pass
    if render_body and render_bodies_static is None:
        mv.add_mesh_seq(body_mesh_seq, progress_bar=progress_bar)
    elif render_body and render_bodies_static is not None:
        mv.add_static_meshes(
            [
                body_mesh_seq[i]
                for i in range(len(body_mesh_seq))
                if i % render_bodies_static == 0
            ]
        )
    if render_joints and render_skeleton:
        mv.add_point_seq(
            joints_seq,
            color=joint_color,
            radius=joint_rad,
            contact_seq=contacts,
            connections=skel_connections,
            connect_color=skel_color,
            vel=joints_vel,
            contact_color=contact_color,
            render_static=render_points_static,
        )
    elif render_joints:
        mv.add_point_seq(
            joints_seq,
            color=joint_color,
            radius=joint_rad,
            contact_seq=contacts,
            vel=joints_vel,
            contact_color=contact_color,
            render_static=render_points_static,
        )

    if vtx_list is not None:
        mv.add_smpl_vtx_list_seq(
            body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015
        )

    if points_seq is not None:
        if torch.is_tensor(points_seq[0]):
            points_seq = [c2c(point_frame) for point_frame in points_seq]
        mv.add_point_seq(
            points_seq,
            color=point_color,
            radius=point_rad,
            vel=points_vel,
            render_static=render_points_static,
        )

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_body:
                xyz_orig = body_mesh_seq[0].vertices[0, :]
            elif render_joints:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = points_seq[0][0, :]

        mv.add_ground(
            ground_plane=ground_plane,
            xyz_orig=xyz_orig,
            color0=ground_color0,
            color1=ground_color1,
            alpha=ground_alpha,
        )

    mv.set_render_settings(
        out_path=out_path,
        wireframe=wireframe,
        RGBA=RGBA,
        single_frame=(
            render_points_static is not None or render_bodies_static is not None
        ),
    )  # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps, start=start, end=end, progress_bar=progress_bar)
    except RuntimeError as err:
        print("Could not render properly with the error: %s" % (str(err)))
    finally:
        # Properly clean up the mesh viewer and its OpenGL context
        try:
            if hasattr(mv, 'viewer') and mv.viewer is not None:
                if hasattr(mv.viewer, 'delete'):
                    mv.viewer.delete()
                elif hasattr(mv.viewer, 'close'):
                    mv.viewer.close()
        except Exception as cleanup_err:
            print(f"Cleanup error (non-fatal): {cleanup_err}")
        del mv
