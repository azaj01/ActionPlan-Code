# Adapted from STMC: src/data/text_motion.py
# Upstream repo: https://github.com/nv-tlabs/stmc
# Source file: https://github.com/nv-tlabs/stmc/blob/main/src/data/text_motion.py

import logging
import os
import codecs as cs
import orjson  # loading faster than json
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion

logger = logging.getLogger(__name__)


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        name: str,
        motion_loader,
        text_encoder,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        subset_size: int | None = None,
        # only during training
        drop_motion_perc: float = 0.15,
        drop_cond: float = 0.10,
        drop_trans: float = 0.5,
        fixed_annotation_index: int | None = None,
        val_same_as_train: bool = False,
    ):
        if tiny:
            split = split + "_tiny"

        path = f"datasets/annotations/{name}"
        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)

        self.text_encoder = text_encoder
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test_lel" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = "train" in split
        self.drop_motion_perc = drop_motion_perc
        self.drop_cond = drop_cond
        self.drop_trans = drop_trans
        self.fixed_annotation_index = fixed_annotation_index

        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        
        # Filter out keyids that don't have corresponding motion files
        # This is important for latent data where some files might be missing
        base_dir = getattr(self.motion_loader, "base_dir", None)
        if base_dir and os.path.exists(base_dir):
            existing_files = set([f.replace(".npy", "") for f in os.listdir(base_dir) if f.endswith(".npy")])
            initial_count = len(self.keyids)
            self.keyids = [
                keyid for keyid in self.keyids 
                if self.annotations[keyid]["path"] in existing_files
            ]
            filtered_count = initial_count - len(self.keyids)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} keyids without corresponding motion files")
        
        # Optionally reduce to a tiny subset for debugging/sanity checks
        if subset_size is not None:
            self.keyids = self.keyids[: int(subset_size)]
        self.nfeats = self.motion_loader.nfeats

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            if self.fixed_annotation_index is not None:
                index = int(self.fixed_annotation_index)
            else:
                index = np.random.randint(len(annotations["annotations"]))

        annotation = annotations["annotations"][index]
        text = annotation["text"]

        drop_motion_perc = None
        load_transition = False
        if self.is_training:
            drop_motion_perc = self.drop_motion_perc
            drop_cond = self.drop_cond
            drop_trans = self.drop_trans
            if drop_cond is not None:
                if np.random.binomial(1, drop_cond) == 1:
                    # uncondionnal
                    text = ""
                    # load a transition
                    if np.random.binomial(1, drop_trans) == 1:
                        load_transition = True

        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
            drop_motion_perc=drop_motion_perc,
            load_transition=load_transition,
        )

        text_encoded = self.text_encoder(text)
        text_uncond_encoded = self.text_encoder("")

        x = motion_x_dict["x"]
        length = motion_x_dict["length"]

        # Check if we're using latent data (t2m_latents)
        is_latent_data = "t2m_latents" in getattr(self.motion_loader, "base_dir", "") or "t2m_latent_frame_text_aligned" in getattr(self.motion_loader, "base_dir", "")

        if is_latent_data:
            # For latent data: only enforce max length of 78
            max_latent_len = 78
            if length >= max_latent_len:
                x = x[:max_latent_len]
                length = max_latent_len
            else:
                x = torch.cat([x, torch.zeros(max_latent_len - length, x.size(1), dtype=x.dtype)], dim=0)
        else:
            # Original behavior for non-latent data
            # Enforce min/max duration and fixed model input length
            # Calculate frame counts based on actual FPS (supports both 20 FPS and 30 FPS data)
            fps = getattr(self.motion_loader, "fps", 20.0)
            desired_len = int(self.max_seconds * fps)  # e.g., 10s * 30fps = 300 frames
            min_len = int(self.min_seconds * fps)      # e.g., 1s * 30fps = 30 frames

            # If too short, extend by repeating the last frame up to min_len
            if length < min_len:
                if length > 0:
                    last = x[-1:].expand(min_len - length, -1)
                    x = torch.cat([x, last], dim=0)
                length = min_len

            # Clamp logical length to desired range for masking
            length = int(max(min_len, min(length, desired_len)))

            # Pad or truncate to exactly desired_len for the model
            if x.size(0) >= desired_len:
                x = x[:desired_len]
            else:
                pad = torch.zeros(desired_len - x.size(0), x.size(1), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)

        output = {
            "x": x,
            "text": text,
            "tx": text_encoded,
            "tx_uncond": text_uncond_encoded,
            "keyid": keyid,
            "length": length,
            "segment_indices": None,
        }
        return output

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():
            path = val["path"]

            # remove humanact12
            # buggy left/right + no SMPL
            if "humanact12" in path:
                continue

            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
