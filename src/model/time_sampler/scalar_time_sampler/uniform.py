# Adapted from SRM: src/model/time_sampler/scalar_time_sampler/uniform.py
# Upstream repo: https://github.com/Chrixtar/SRM
# Source file: https://github.com/Chrixtar/SRM/blob/main/src/model/time_sampler/scalar_time_sampler/uniform.py

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import device, Tensor

from .scalar_time_sampler import ScalarTimeSampler, ScalarTimeSamplerCfg


@dataclass
class UniformCfg(ScalarTimeSamplerCfg):
    name: Literal["uniform"]


class Uniform(ScalarTimeSampler[UniformCfg]):
    
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu"
    ) -> Tensor:
        return torch.rand(shape, device=device)
