# Adapted from SRM: src/model/time_sampler/two_stage_time_sampler.py
# Upstream repo: https://github.com/Chrixtar/SRM
# Source file: https://github.com/Chrixtar/SRM/blob/main/src/model/time_sampler/two_stage_time_sampler.py

from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar

from .scalar_time_sampler import (
    ScalarTimeSampler, 
    ScalarTimeSamplerCfg, 
    get_scalar_time_sampler,
    UniformCfg
)
from .time_sampler import TimeSampler, TimeSamplerCfg


@dataclass
class TwoStageTimeSamplerCfg(TimeSamplerCfg):
    scalar_time_sampler: ScalarTimeSamplerCfg = field(default_factory=lambda: UniformCfg("uniform"))


T = TypeVar("T", bound=TwoStageTimeSamplerCfg)


class TwoStageTimeSampler(TimeSampler[T], ABC):
    scalar_time_sampler: ScalarTimeSampler

    def __init__(
        self,
        cfg: T,
        resolution: tuple[int, int],
    ) -> None:
        super(TwoStageTimeSampler, self).__init__(cfg, resolution)
        self.scalar_time_sampler = get_scalar_time_sampler(cfg.scalar_time_sampler)
