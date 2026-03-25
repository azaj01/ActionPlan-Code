# Adapted from STMC: src/renderer/humor.py
# Upstream repo: https://github.com/nv-tlabs/stmc
# Source file: https://github.com/nv-tlabs/stmc/blob/main/src/renderer/humor.py

import numpy as np
import torch
from .humor_render_tools.tools import viz_smpl_seq
from smplx.utils import Struct
from .video import Video
import os
import shutil
from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import Process

# os.environ["PYOPENGL_PLATFORM"] = "egl"


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
FACE_PATH = os.path.join(THIS_FOLDER, "humor_render_tools/smplh.faces")
FACES = torch.from_numpy(np.int32(np.load(FACE_PATH)))


class HumorRenderer:
    def __init__(self, fps=20.0, convert_yup_to_zup=False, **kwargs):
        """
        Initialize the HumorRenderer.
        
        Args:
            fps: Frames per second for rendering
            convert_yup_to_zup: If True, convert from Y-up to Z-up coordinate system.
                               This should only be True for latent models (272-dim).
                               Standard SMPL models already output in the correct space.
            **kwargs: Additional arguments passed to the render function
        """
        self.kwargs = kwargs
        self.fps = fps
        self.convert_yup_to_zup = convert_yup_to_zup

    def __call__(self, vertices, output, **kwargs):
        params = self.kwargs | kwargs
        fps = self.fps
        if "fps" in params:
            fps = params.pop("fps")
        # Pass convert_yup_to_zup from instance, but allow override via kwargs
        convert_yup_to_zup = params.pop("convert_yup_to_zup", self.convert_yup_to_zup)
        render(vertices, output, fps, convert_yup_to_zup=convert_yup_to_zup, **params)


def render(vertices, out_path, fps, progress_bar=tqdm, convert_yup_to_zup=False, **kwargs):
    """
    Render SMPL vertices to video.
    
    Args:
        vertices: [T, V, 3] SMPL vertices
        out_path: Output video path
        fps: Frames per second
        convert_yup_to_zup: If True, convert from Y-up (latent model output) to Z-up (visualization).
                           This should only be True for latent models (272-dim).
                           Standard SMPL models already output in the correct coordinate space.
    """
    # Convert from Y-up (SMPL/272-dim native) to Z-up (visualization)
    # 
    # Coordinate system definitions:
    # Y-up: x=right, y=up, z=forward
    # Z-up: x=right, y=forward, z=up
    #
    # Transformation: [-x, z, y]
    # This maps coordinates by:
    # - Negating x (flips left/right to match visualization conventions)
    # - Mapping old z (forward) to new y (forward)
    # - Mapping old y (up) to new z (up)
    if convert_yup_to_zup:
        x = vertices[..., 0]
        y = vertices[..., 1]
        z = vertices[..., 2]
        # Apply coordinate transformation: [x, y, z]_yup -> [-x, z, y]_zup
        vertices = np.stack([-x, z, y], axis=-1)
    
    # Put the vertices at the floor level
    ground = vertices[..., 2].min()
    vertices[..., 2] -= ground

    import pyrender

    # remove title if it exists
    kwargs.pop("title", None)

    # vertices: SMPL-H vertices
    # verts = np.load("interval_2_verts.npy")
    out_folder = os.path.splitext(out_path)[0]

    # Ensure a clean frame folder for this render to avoid leftover frames
    # from previous renders causing spurious extra motion or wrong overlays
    try:
        if os.path.isdir(out_folder):
            shutil.rmtree(out_folder, ignore_errors=True)
        os.makedirs(out_folder, exist_ok=True)
    except Exception:
        # If cleanup fails, proceed; downstream may still overwrite frames
        pass

    verts = torch.from_numpy(vertices)
    body_pred = Struct(v=verts, f=FACES)

    # ==================== Camera and Ground Settings ====================
    # Match the sequential visualization style
    
    # enable follow_camera by default 
    if "follow_camera" not in kwargs:
        kwargs["follow_camera"] = True
    
    # Solid light gray ground color (same as sequential visualization)
    if "ground_color0" not in kwargs:
        kwargs["ground_color0"] = [0.85, 0.85, 0.85]
    if "ground_color1" not in kwargs:
        kwargs["ground_color1"] = [0.85, 0.85, 0.85]

    # out_folder, body_pred, start, end, fps, kwargs = args
    viz_smpl_seq(
        pyrender, out_folder, body_pred, fps=fps, progress_bar=progress_bar, **kwargs
    )

    video = Video(out_folder, fps=fps)
    video.save(out_path)


def render_offset(args):
    import pyrender

    out_folder, body_pred, start, end, fps, kwargs = args
    viz_smpl_seq(
        pyrender, out_folder, body_pred, start=start, end=end, fps=fps, **kwargs
    )
    return 0


def render_multiprocess(vertices, out_path, fps, **kwargs):
    # WIP: does not work yet
    import ipdb

    ipdb.set_trace()
    # remove title if it exists
    kwargs.pop("title", None)

    # vertices: SMPL-H vertices
    # verts = np.load("interval_2_verts.npy")
    out_folder = os.path.splitext(out_path)[0]

    verts = torch.from_numpy(vertices)
    body_pred = Struct(v=verts, f=FACES)

    # faster rendering
    # by rendering part of the sequence in parallel
    # still work in progress, use one process for now
    n_processes = 1

    verts_lst = np.array_split(verts, n_processes)
    len_split = [len(x) for x in verts_lst]
    starts = [0] + np.cumsum([x for x in len_split[:-1]]).tolist()
    ends = np.cumsum([x for x in len_split]).tolist()
    out_folders = [out_folder for _ in range(n_processes)]
    fps_s = [fps for _ in range(n_processes)]
    kwargs_s = [kwargs for _ in range(n_processes)]
    body_pred_s = [body_pred for _ in range(n_processes)]

    arguments = [out_folders, body_pred_s, starts, ends, fps_s, kwargs_s]
    # sanity
    # lst = [verts[start:end] for start, end in zip(starts, ends)]
    # assert (torch.cat(lst) == verts).all()

    processes = []
    for _, args in zip(range(n_processes), zip(*arguments)):
        process = Process(target=render_offset, args=(args,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    if False:
        # start 4 worker processes
        with Pool(processes=n_processes) as pool:
            # print "[0, 1, 4,..., 81]"
            # print same numbers in arbitrary order
            print(f"0/{n_processes} rendered")
            i = 0
            for _ in pool.imap_unordered(render_offset, zip(*arguments)):
                i += 1
                print(f"i/{n_processes} rendered")

    video = Video(out_folder, fps=fps)
    video.save(out_path)
