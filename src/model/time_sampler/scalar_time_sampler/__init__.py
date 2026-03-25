# Adapted from SRM: src/model/time_sampler/scalar_time_sampler/__init__.py
# Upstream repo: https://github.com/Chrixtar/SRM
# Source file: https://github.com/Chrixtar/SRM/blob/main/src/model/time_sampler/scalar_time_sampler/__init__.py

from .scalar_time_sampler import ScalarTimeSampler
from .uniform import Uniform, UniformCfg


SCALAR_TIME_SAMPLER = {
    "uniform": Uniform
}


ScalarTimeSamplerCfg = UniformCfg


def get_scalar_time_sampler(
    cfg: ScalarTimeSamplerCfg
) -> ScalarTimeSampler:
    return SCALAR_TIME_SAMPLER[cfg.name](cfg)
