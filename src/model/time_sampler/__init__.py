from .time_sampler import TimeSampler
from .motion_time_sampler import MotionTimeSampler, MotionTimeSamplerCfg

TimeSamplerCfg = MotionTimeSamplerCfg


def get_time_sampler(
    cfg: TimeSamplerCfg, resolution: tuple[int, int]
) -> TimeSampler:
    if cfg.name != "motion":
        raise ValueError(f"Only 'motion' time sampler is supported, got: {cfg.name}")
    return MotionTimeSampler(cfg, resolution)
