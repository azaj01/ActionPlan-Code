# Adapted from STMC: src/data/collate.py
# Upstream repo: https://github.com/nv-tlabs/stmc
# Source file: https://github.com/nv-tlabs/stmc/blob/main/src/data/collate.py

import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    # Keys that need special handling (not default_collate-able)
    skip_keys = {"x", "tx", "segment_indices", "tx_uncond"}
    other_keys = [key for key in keys if key not in skip_keys]
    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}

    x = collate_tensor_with_padding([x["x"] for x in lst_elements])
    if device is not None:
        x = x.to(device)

    batch["x"] = x

    if "length" in batch:
        # Build mask to match the padded/truncated width of x
        lengths = torch.as_tensor(batch["length"], device=x.device)
        T = x.size(1)
        batch["mask"] = torch.arange(T, device=x.device).expand(len(lengths), T) < lengths.unsqueeze(1)

    # text embeddings
    if "tx" in keys:
        assert "x" in one_el["tx"]
        assert "length" in one_el["tx"]
        tx_x = collate_tensor_with_padding([x["tx"]["x"] for x in lst_elements])
        tx_length = default_collate([x["tx"]["length"] for x in lst_elements])
        tx_mask = length_to_mask(tx_length, device=tx_x.device)
        batch["tx"] = {"x": tx_x, "length": tx_length, "mask": tx_mask}

    if "tx_uncond" in keys:
        # only one is enough
        batch["tx_uncond"] = one_el["tx_uncond"]

    # Semantic patching: segment_indices is a list of (Tensor | None), variable length
    if "segment_indices" in keys:
        batch["segment_indices"] = [x["segment_indices"] for x in lst_elements]

    return batch


def collate_text_motion_actionplan_merged(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    """Collate for ActionPlanMergedDataset: same as collate_text_motion + has_text_latent as tensor."""
    batch = collate_text_motion(lst_elements, device=device)
    if "has_text_latent" in lst_elements[0]:
        batch["has_text_latent"] = torch.tensor(
            [x["has_text_latent"] for x in lst_elements],
            dtype=torch.bool,
            device=batch["x"].device if device is None else device,
        )
    return batch





### TMR ###
def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = length_to_mask(length, device=x.device)
    batch = {"x": x, "length": length, "mask": mask}
    return batch
