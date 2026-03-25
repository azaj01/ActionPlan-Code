from dataclasses import dataclass
from math import prod
from typing import Literal

import torch
from torch import device, Tensor
from torch.distributions.beta import Beta

from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class MotionTimeSamplerCfg(TwoStageTimeSamplerCfg):
    name: Literal["motion"]
    beta_sharpness: float = 1.0
    per_feature: bool = False


class MotionTimeSampler(TwoStageTimeSampler[MotionTimeSamplerCfg]):

    def __init__(
        self,
        cfg: MotionTimeSamplerCfg,
        resolution: tuple[int, int],
    ) -> None:
        super(MotionTimeSampler, self).__init__(cfg, resolution)
        self.dim = prod(resolution)
        self._betas: dict[int, Beta] = {}
        self._init_betas(self.dim)

    def _init_betas(self, dim: int) -> None:
        if dim > 1 and dim not in self._betas:
            a = b = (dim - 1 - (dim % 2)) ** 1.05 * self.cfg.beta_sharpness
            self._betas[dim] = Beta(a, b)
            half_dim = dim // 2
            self._init_betas(half_dim)
            self._init_betas(dim - half_dim)

    def _get_uniform_l1_conditioned_vector_list(
        self,
        l1_norms: Tensor,
        dim: int,
    ) -> list[Tensor]:
        if dim == 1:
            return [l1_norms]

        device_ = l1_norms.device
        half_cells = dim // 2

        max_first_contribution = l1_norms.clamp(max=half_cells)
        max_second_contribution = l1_norms.clamp(max=dim - half_cells)
        min_first_contribution = (l1_norms - max_second_contribution).clamp_(min=0)

        random_matrix = self._betas[dim].sample((l1_norms.shape[0],)).to(device=device_)
        ranges = max_first_contribution - min_first_contribution

        first_contribution = min_first_contribution + ranges * random_matrix
        second_contribution = l1_norms - first_contribution

        return self._get_uniform_l1_conditioned_vector_list(first_contribution, half_cells) \
            + self._get_uniform_l1_conditioned_vector_list(second_contribution, dim - half_cells)

    def _sample_time_matrix(
        self,
        l1_norms: Tensor,
        dim: int,
    ) -> Tensor:
        vector_list = self._get_uniform_l1_conditioned_vector_list(l1_norms, dim)
        t = torch.stack(vector_list, dim=1)
        idx = torch.rand_like(t).argsort()
        t = t.gather(1, idx)
        return t

    def get_time_with_mean(self, mean: Tensor) -> Tensor:
        shape = mean.shape
        l1_norms = mean.flatten() * self.dim
        t = self._sample_time_matrix(l1_norms, self.dim)
        return t.view(*shape, *self.resolution)

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> Tensor:
        # Sample scalar t_bar and generate per-element t by recursive l1-conditioned splitting
        mean = self.scalar_time_sampler((batch_size, num_samples), device)
        return self.get_time_with_mean(mean)
