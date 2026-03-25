"""
ActionPlan merged dataset: loads t2m_latent_frame_text_aligned (32-dim) where available,
otherwise t2m_latents (16-dim padded to 32). Flags has_text_latent for loss masking.
"""

import logging
import os
import codecs as cs
import orjson

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion_actionplan_merged

logger = logging.getLogger(__name__)


def _read_split(path: str, split: str) -> list[str]:
    split_file = os.path.join(path, "splits", split + ".txt")
    ids = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            ids.append(line.strip())
    return ids


def _read_train_frame_level_subset_ids(path: str) -> set[str]:
    """Read splits/train_frame_level_subset.txt (IDs that have train_frame_level_subset data)."""
    fpath = os.path.join(path, "splits", "train_frame_level_subset.txt")
    if not os.path.exists(fpath):
        return set()
    with cs.open(fpath, "r") as f:
        return set(line.strip() for line in f.readlines() if line.strip())


def _load_annotations(path: str, name: str = "annotations.json") -> dict:
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


def _filter_annotations(annotations: dict, min_seconds: float, max_seconds: float) -> dict:
    filtered = {}
    for key, val in list(annotations.items()):
        if "humanact12" in val.get("path", ""):
            continue
        annots = val.get("annotations", [])
        filtered_annots = [
            a for a in annots
            if max_seconds >= (a["end"] - a["start"]) >= min_seconds
        ]
        if filtered_annots:
            val = dict(val)
            val["annotations"] = filtered_annots
            filtered[key] = val
    return filtered


class ActionPlanMergedDataset(Dataset):
    """Dataset that loads t2m_latent_frame_text_aligned (32-dim) or t2m_latents (16-dim padded to 32)."""

    def __init__(
        self,
        name: str,
        motion_loader_t2m_latent_frame_text_aligned,
        motion_loader_t2m_latents,
        text_encoder,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        tiny: bool = False,
        subset_size: int | None = None,
        drop_motion_perc: float = 0.15,
        drop_cond: float = 0.10,
        drop_trans: float = 0.5,
        fixed_annotation_index: int | None = None,
        val_same_as_train: bool = False,
    ):
        if tiny:
            split = split + "_tiny"

        path = os.path.join("datasets", "annotations", name)
        self.collate_fn = collate_text_motion_actionplan_merged
        self.split = split
        self.keyids = _read_split(path, split)
        self.train_frame_level_subset_ids = _read_train_frame_level_subset_ids(path)
        if not self.train_frame_level_subset_ids :
            raise ValueError("train_frame_level_subset.txt not found")

        self.text_encoder = text_encoder
        self.motion_loader_uni = motion_loader_t2m_latent_frame_text_aligned
        self.motion_loader_t2m = motion_loader_t2m_latents

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.annotations = _load_annotations(path)
        if "test_lel" not in split:
            self.annotations = _filter_annotations(
                self.annotations, min_seconds, max_seconds
            )

        self.is_training = "train" in split
        self.drop_motion_perc = drop_motion_perc
        self.drop_cond = drop_cond
        self.drop_trans = drop_trans
        self.fixed_annotation_index = fixed_annotation_index

        self.keyids = [k for k in self.keyids if k in self.annotations]

        # Keep only keyids that have at least one motion source (t2m_latent_frame_text_aligned or t2m_latents)
        uni_dir = getattr(self.motion_loader_uni, "base_dir", None)
        t2m_dir = getattr(self.motion_loader_t2m, "base_dir", None)
        existing_uni = set()
        existing_t2m = set()
        if uni_dir and os.path.exists(uni_dir):
            existing_uni = {f.replace(".npy", "") for f in os.listdir(uni_dir) if f.endswith(".npy")}
        if t2m_dir and os.path.exists(t2m_dir):
            existing_t2m = {f.replace(".npy", "") for f in os.listdir(t2m_dir) if f.endswith(".npy")}

        def has_motion(keyid: str) -> bool:
            path_id = self.annotations[keyid]["path"]
            # Use the loader that will actually be used at load time
            if path_id in self.train_frame_level_subset_ids:
                return path_id in existing_uni
            else:
                return path_id in existing_t2m

        initial = len(self.keyids)
        self.keyids = [k for k in self.keyids if has_motion(k)]
        if initial != len(self.keyids):
            logger.info("Filtered out %d keyids without motion files", initial - len(self.keyids))

        if subset_size is not None:
            self.keyids = self.keyids[: int(subset_size)]

        self.nfeats = 32

        if preload:
            for _ in tqdm(self, desc="Preloading actionplan merged dataset"):
                continue

    def __len__(self) -> int:
        return len(self.keyids)

    def __getitem__(self, index: int) -> dict:
        return self.load_keyid(self.keyids[index])

    def load_keyid(self, keyid: str) -> dict:
        annotations = self.annotations[keyid]
        path_id = annotations["path"]

        idx = 0
        if self.is_training:
            if self.fixed_annotation_index is not None:
                idx = int(self.fixed_annotation_index)
            else:
                idx = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][idx]
        text = annotation["text"]

        drop_motion_perc = None
        load_transition = False
        if self.is_training:
            drop_motion_perc = self.drop_motion_perc
            if np.random.binomial(1, self.drop_cond) == 1:
                text = ""
                if np.random.binomial(1, self.drop_trans) == 1:
                    load_transition = True

        use_frame_level = path_id in self.train_frame_level_subset_ids
        try:
            if use_frame_level:
                motion_x_dict = self.motion_loader_uni(
                    path=path_id,
                    start=annotation["start"],
                    end=annotation["end"],
                    drop_motion_perc=drop_motion_perc,
                    load_transition=load_transition,
                )
                x = motion_x_dict["x"]
                length = motion_x_dict["length"]
                has_text_latent = True
            else:
                motion_x_dict = self.motion_loader_t2m(
                    path=path_id,
                    start=annotation["start"],
                    end=annotation["end"],
                    drop_motion_perc=drop_motion_perc,
                    load_transition=load_transition,
                )
                x = motion_x_dict["x"]
                length = motion_x_dict["length"]
                # Pad 16 -> 32 with zeros for text dims
                x = torch.cat([x, torch.zeros(x.size(0), 16, dtype=x.dtype)], dim=-1)
                has_text_latent = False
        except FileNotFoundError:
                raise

        # Latent length handling: max 78 frames, pad or truncate
        max_latent_len = 78
        if length >= max_latent_len:
            x = x[:max_latent_len]
            length = max_latent_len
        else:
            x = torch.cat([
                x,
                torch.zeros(max_latent_len - length, x.size(1), dtype=x.dtype),
            ], dim=0)

        text_encoded = self.text_encoder(text)
        text_uncond_encoded = self.text_encoder("")

        return {
            "x": x,
            "text": text,
            "tx": text_encoded,
            "tx_uncond": text_uncond_encoded,
            "keyid": keyid,
            "length": length,
            "has_text_latent": has_text_latent,
            "segment_indices": None,
        }
