# Adapted from STMC: src/data/motion.py
# Upstream repo: https://github.com/nv-tlabs/stmc
# Source file: https://github.com/nv-tlabs/stmc/blob/main/src/data/motion.py

import os
import torch
import numpy as np
import random


class AMASSMotionLoader:
    def __init__(
        self,
        base_dir,
        fps,
        disable: bool = False,
        nfeats=None,
        umin_s=0.5,
        umax_s=3.0,
        channel_first: bool = True,
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.disable = disable
        self.nfeats = nfeats
        # If True, files are (C, L); transpose to (L, C) after load so first dim is time
        self.channel_first = channel_first

        # Auto-detect if we're using absolute root motion data
        self.abs_root = "_abs" in base_dir

        # unconditional, sampling the duration from [umin, umax]
        self.umin = int(self.fps * umin_s)
        assert self.umin > 0
        self.umax = int(self.fps * umax_s)

    def __call__(self, path, start, end, drop_motion_perc=None, load_transition=False):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        # load the motion
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + ".npy")
            if not os.path.exists(motion_path):
                raise FileNotFoundError(
                    f"Motion file not found: {motion_path}\n"
                    f"  base_dir: {self.base_dir}\n"
                    f"  path: {path}\n"
                    f"  Check if the file exists or if the path in annotations matches the actual file names."
                )
            motion = np.load(motion_path)
            motion = torch.from_numpy(motion).to(torch.float)
            if self.channel_first:
                motion = motion.T  # (C, L) -> (L, C)
            self.motions[path] = motion

        if load_transition:
            motion = self.motions[path]
            # take a random crop
            duration = random.randint(self.umin, min(self.umax, len(motion)))
            # random start
            start = random.randint(0, len(motion) - duration)
            motion = motion[start : start + duration]
        else:
            begin = int(start * self.fps)
            end = int(end * self.fps)
            motion = self.motions[path][begin:end]

            # crop max X% of the motion randomly beginning and end
            if drop_motion_perc is not None:
                max_frames_to_drop = int(len(motion) * drop_motion_perc)
                # randomly take a number of frames to drop
                n_frames_to_drop = random.randint(0, max_frames_to_drop)

                # split them between left and right
                n_frames_left = random.randint(0, n_frames_to_drop)
                n_frames_right = n_frames_to_drop - n_frames_left

                # crop the motion safely
                # clamp total frames to drop to keep at least one frame
                n_frames_to_drop = min(n_frames_to_drop, max(0, len(motion) - 1))
                n_frames_left = min(n_frames_left, n_frames_to_drop)
                n_frames_right = min(n_frames_right, n_frames_to_drop - n_frames_left)

                start_idx = n_frames_left
                end_idx = len(motion) - n_frames_right

                # ensure valid, non-empty range
                if end_idx <= start_idx:
                    end_idx = min(len(motion), start_idx + 1)
                if end_idx <= 0:
                    start_idx = 0
                    end_idx = 1

                motion = motion[start_idx:end_idx]

        x_dict = {"x": motion, "length": len(motion)}
        return x_dict
