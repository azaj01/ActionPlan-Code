# Adapted from SRM: src/model/time_sampler/time_sampler.py
# Upstream repo: https://github.com/Chrixtar/SRM
# Source file: https://github.com/Chrixtar/SRM/blob/main/src/model/time_sampler/time_sampler.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import prod
from typing import Generic, TypeVar

import torch
from torch import device, Tensor
from torch.nn import functional as F


@dataclass
class HistogramPdfEstimatorCfg:
    num_bins: int = 1000
    blur_kernel_size: int = 5
    blur_kernel_sigma: float = 0.2


class HistogramPdfEstimator:
    """Estimates the density of a set of samples and returns the inverse of the density as weights."""

    def __init__(
        self,
        initial_samples: Tensor,
        cfg: HistogramPdfEstimatorCfg,
    ) -> None:
        self.cfg = cfg
        # Note: histogram is stored on the device of initial_samples
        # and will be moved to the correct device in __call__ if needed
        self.histogram = self.get_smooth_density_histogram(initial_samples)
        assert self.histogram.min().item() > 0.001, \
            "Histogram too inaccurate, please use a different time_sampler"

    def get_gaussian_1d_kernel(self, device: device | str) -> Tensor:
        """Create a 1D Gaussian kernel on the specified device."""
        if self.cfg.blur_kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        # Create a range of values centered at 0
        center = self.cfg.blur_kernel_size // 2
        x = torch.arange(-center, center + 1, dtype=torch.float32, device=device)

        # Compute the Gaussian function
        kernel = torch.exp(-0.5 * (x / self.cfg.blur_kernel_sigma) ** 2)

        # Normalize the kernel to ensure sum equals 1
        kernel /= kernel.sum()
        return kernel

    def get_smooth_density_histogram(
        self, 
        vals: Tensor
    ) -> Tensor:
        """Compute smoothed density histogram, deriving device from input tensor."""
        assert vals.min() >= 0, "Timesteps must be nonnegative"
        assert vals.max() <= 1, "Timesteps must be less or equal than 1"
        
        # Derive device from input tensor
        input_device = vals.device
        histogram_torch = torch.histc(vals, self.cfg.num_bins, min=0, max=1).to(
            input_device
        )

        kernel = self.get_gaussian_1d_kernel(input_device)

        # Reflective padding to avoid edge effects in convolution
        padded_hist = F.pad(
            histogram_torch.unsqueeze(0).unsqueeze(0),
            (self.cfg.blur_kernel_size // 2, self.cfg.blur_kernel_size // 2),
            mode="reflect",
        )
        histogram_torch_conv = F.conv1d(
            padded_hist, kernel.unsqueeze(0).unsqueeze(0)
        ).to(input_device)

        # remove unnecessary dimensions and normalize to pdf
        return histogram_torch_conv.squeeze() / histogram_torch_conv.mean()

    def __call__(
        self, 
        t: Tensor
    ) -> Tensor:
        """Look up histogram values, moving histogram to input device if needed."""
        # Move histogram to input device if necessary (for DDP compatibility)
        if self.histogram.device != t.device:
            self.histogram = self.histogram.to(t.device)
        
        bin_ids = (t * self.cfg.num_bins).long()
        bin_ids.clamp_(0, self.cfg.num_bins - 1)
        return self.histogram[bin_ids]


@dataclass
class TimeSamplerCfg:
    name: str
    histogram_pdf_estimator: HistogramPdfEstimatorCfg = field(
        default_factory=HistogramPdfEstimatorCfg
    )
    num_normalization_samples: int = 80000
    eps: float = 1e-6
    add_zeros: bool = False


T = TypeVar("T", bound=TimeSamplerCfg)


class TimeSampler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        resolution: tuple[int, int],
    ) -> None:
        self.cfg = cfg
        self.resolution = resolution
        self.dim = prod(self.resolution)

    @abstractmethod
    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> Tensor:
        pass

    def get_normalization_samples(
        self, 
        device: device | str
    ) -> Tensor:
        return self.get_time(
            self.cfg.num_normalization_samples, device=device
        ).flatten()

    def get_normalization_weights(
        self, 
        t: Tensor
    ) -> Tensor:
        if self.cfg.histogram_pdf_estimator is None:
            return torch.ones_like(t)

        if not hasattr(self, "histogram_pdf_estimator"):
            self.histogram_pdf_estimator = HistogramPdfEstimator(
                self.get_normalization_samples(t.device),
                self.cfg.histogram_pdf_estimator,
            )

        shape = t.shape
        probs = self.histogram_pdf_estimator(t.flatten())
        weights = (1 + self.cfg.eps) / (probs + self.cfg.eps)
        return weights.view(shape)

    def __call__(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        t = self.get_time(batch_size, num_samples, device)
        weights = self.get_normalization_weights(t)

        ############        ADD ZEROS       ###########
        if False:
            t = t.flatten(-2).contiguous()
            weights = weights.flatten(-2).contiguous()
            zero_ratios = torch.rand((batch_size,), device=device)
            zero_mask = torch.linspace(1/self.dim, 1, self.dim, device=device) < zero_ratios[:, None, None]
            idx = torch.rand_like(t).argsort(dim=-1)
            t[zero_mask.squeeze()] = 0
            weights[zero_mask.squeeze()] = 0
            t = t.gather(-1, idx).reshape(batch_size, -1, *self.resolution)
            weights = weights.gather(-1, idx).reshape(batch_size, -1, *self.resolution)    



        return t, weights
