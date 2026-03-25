import logging
from collections import defaultdict
from math import log
from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm

from .utils import masked
from .time_sampler import MotionTimeSamplerCfg, get_time_sampler
from ..data.collate import length_to_mask
from ..tools.eval_runner import infer_is_latent_model, run_validation_eval


logger = logging.getLogger(__name__)


class RectifiedFlowDiffusion(LightningModule):
    """Rectified-flow backbone with a diffusion-compatible sampler API."""

    name = "rectified_flow"
    uses_rectified_flow = True

    def __init__(
        self,
        denoiser,
        timesteps,
        motion_normalizer,
        text_normalizer,
        lr: float = 2e-4,
        prediction: str = "v",
        time_sampler_cfg: MotionTimeSamplerCfg = None,
        per_feature: bool = False,
        inference_length: Optional[int] = None,
        sigma_weight: float = 0.0,
        predict_sigma: bool = False,
        variance_mode: str = "fixed_large",
        variance_alpha: float = 1.0,
        sampling_temperature: float = 1.0,
        stochastic_sampling: bool = True,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        shared_timestep_per_frame: bool = False,
        t_zero_clean: bool = True,
        diffusion_type: str = "rectified_flow",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.motion_normalizer = motion_normalizer
        self.text_normalizer = text_normalizer
        self.inference_length = inference_length
        self.lr = float(lr)
        self.prediction = str(prediction)
        self.diffusion_type = str(diffusion_type)
        self.per_feature = bool(per_feature)
        self.sigma_weight = float(sigma_weight)
        self.predict_sigma = bool(predict_sigma)
        self.variance_mode = str(variance_mode)
        self.variance_alpha = float(variance_alpha)
        self.sampling_temperature = float(sampling_temperature)
        self.stochastic_sampling = bool(stochastic_sampling)
        self.shared_timestep_per_frame = bool(shared_timestep_per_frame)
        self.t_zero_clean = bool(t_zero_clean)
        self.reconstruction_loss = torch.nn.MSELoss(reduction="none")

        if self.prediction not in {"v"}:
            raise ValueError(
                f"RectifiedFlowDiffusion only supports prediction='v', got: {self.prediction}"
            )
        if self.variance_mode != "fixed_large":
            raise ValueError(
                f"RectifiedFlowDiffusion currently supports only variance_mode='fixed_large', got: {self.variance_mode}"
            )

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        if self.use_ema:
            self.ema_denoiser = AveragedModel(
                self.denoiser,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
            )
            for param in self.ema_denoiser.parameters():
                param.requires_grad_(False)
        else:
            self.ema_denoiser = None

        if time_sampler_cfg is None:
            time_sampler_cfg = MotionTimeSamplerCfg(name="motion", per_feature=self.per_feature)
        self.time_sampler_cfg = time_sampler_cfg
        self.time_sampler = None

        self._saved_losses = defaultdict(list)
        self.losses = []

    @property
    def device(self):
        return next(self.parameters()).device

    def configure_optimizers(self):
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def _init_time_sampler(self, x_shape):
        n_frames, n_features = x_shape[1], x_shape[2]
        resolution = (n_frames, n_features) if self.time_sampler_cfg.per_feature else (n_frames,)
        need_rebuild = (
            self.time_sampler is None
            or getattr(self.time_sampler, "resolution", None) != resolution
        )
        if need_rebuild:
            self.time_sampler = get_time_sampler(self.time_sampler_cfg, resolution)

    def _to_tau(self, t, like_tensor):
        return t.to(dtype=like_tensor.dtype).clamp(0.0, 1.0)

    def prepare_tx_emb(self, tx_emb):
        if "mask" not in tx_emb:
            tx_emb["mask"] = length_to_mask(tx_emb["length"], device=self.device)
        return {
            "x": masked(self.text_normalizer(tx_emb["x"]), tx_emb["mask"]),
            "length": tx_emb["length"],
            "mask": tx_emb["mask"],
        }

    def flow_forward_sample(self, xstart, t, noise=None):
        if noise is None:
            noise = torch.randn_like(xstart)
        tau = self._to_tau(t, xstart)
        while tau.ndim < xstart.ndim:
            tau = tau.unsqueeze(-1)
        return (1.0 - tau) * xstart + tau * noise

    def q_sample(self, xstart, t, noise=None):
        return self.flow_forward_sample(xstart, t, noise=noise)

    def _get_denoiser(self, use_ema: bool = True):
        if use_ema and self.use_ema and self.ema_denoiser is not None:
            return self.ema_denoiser
        return self.denoiser

    def _guided_model_output(self, xt, y, t, use_ema: bool = True):
        if t.dim() == 1:
            t = t.unsqueeze(-1).expand(t.shape[0], xt.shape[1])
        t_model = self._to_tau(t, xt)
        denoiser = self._get_denoiser(use_ema=use_ema)
        out_cond = denoiser(xt, y, t_model)
        if isinstance(out_cond, tuple):
            v_cond, logvar_cond = out_cond
        else:
            v_cond, logvar_cond = out_cond, None
        guidance_weight = y["infos"].get("guidance_weight", 1.0)
        if guidance_weight == 1.0:
            return v_cond, logvar_cond

        y_uncond = y.copy()
        y_uncond["tx"] = y_uncond["tx_uncond"]
        out_uncond = denoiser(xt, y_uncond, t_model)
        if isinstance(out_uncond, tuple):
            v_uncond, logvar_uncond = out_uncond
        else:
            v_uncond, logvar_uncond = out_uncond, None

        v = v_uncond + guidance_weight * (v_cond - v_uncond)
        if logvar_cond is None or logvar_uncond is None:
            return v, logvar_cond
        logvar = logvar_uncond + guidance_weight * (logvar_cond - logvar_uncond)
        return v, logvar

    def _a(self, tau):
        return 1.0 - tau

    def _b(self, tau):
        return tau

    def _sigma_large(self, tau, tau_next, alpha):
        a_t = self._a(tau)
        a_t_next = self._a(tau_next)
        b_t = self._b(tau)
        b_t_next = self._b(tau_next)
        denom = (a_t_next * b_t).clamp(min=1e-8)
        ratio = (a_t * b_t_next) / denom
        inside = (1.0 - ratio.pow(2)).clamp(min=0.0)
        sigma = alpha * b_t * torch.sqrt(inside)
        sigma = torch.where((tau <= 0.0) | (tau_next >= 1.0), torch.zeros_like(sigma), sigma)
        return sigma

    def _gamma(self, tau, tau_next, alpha):
        a_t = self._a(tau)
        a_t_next = self._a(tau_next)
        b_t = self._b(tau)
        b_t_next = self._b(tau_next)
        denom = (a_t_next * b_t).clamp(min=1e-8)
        ratio = (a_t * b_t_next) / denom
        inside = (1.0 - alpha ** 2 * (1.0 - ratio.pow(2))).clamp(min=0.0)
        gamma = b_t_next * torch.sqrt(inside)
        gamma = torch.where(tau <= 0.0, torch.zeros_like(gamma), gamma)
        gamma = torch.where(tau_next >= 1.0, b_t_next, gamma)
        return gamma

    def _transition_stats(self, xt, v_theta, sigma_theta, t, t_next, alpha, temperature):
        tau = self._to_tau(t, xt)
        tau_next = self._to_tau(t_next, xt)
        a_t = self._a(tau)
        a_t_next = self._a(tau_next)
        gamma = self._gamma(tau, tau_next, alpha)
        sigma_base = self._sigma_large(tau, tau_next, alpha)

        while a_t.ndim < xt.ndim:
            a_t = a_t.unsqueeze(-1)
            a_t_next = a_t_next.unsqueeze(-1)
            gamma = gamma.unsqueeze(-1)
            sigma_base = sigma_base.unsqueeze(-1)

        mean = a_t_next * (xt - tau.unsqueeze(-1) * v_theta) + (xt + a_t * v_theta) * gamma
        var = sigma_base.pow(2)
        if sigma_theta is not None:
            coef = a_t_next * tau.unsqueeze(-1) - a_t * gamma
            var = var + (coef * sigma_theta).pow(2)
        if temperature != 1.0:
            var = (temperature ** 2) * var
        std = torch.sqrt(var.clamp(min=0.0))
        return mean, std

    def ode_step(self, xt, y, t, t_next=None, dt=None, use_ema: bool = True):
        if t.dim() == 1:
            t = t.unsqueeze(-1).expand(t.shape[0], xt.shape[1])
        v, _ = self._guided_model_output(xt, y, t, use_ema=use_ema)
        tau = self._to_tau(t, xt)
        if t_next is not None:
            tau_next = self._to_tau(t_next, xt)
            dt = (tau - tau_next).clamp(min=0.0)
        elif dt is None:
            dt = torch.full_like(tau, 1.0 / float(self.timesteps))
        while dt.ndim < xt.ndim:
            dt = dt.unsqueeze(-1)
        x_next = xt - v * dt
        return x_next, v

    def p_sample(self, xt, y, t, t_next=None, stochastic: Optional[bool] = None, use_ema: bool = True):
        mean, std, v = self.p_distribution(
            xt, y, t, t_next=t_next, return_output=True, use_ema=use_ema
        )
        if stochastic is None:
            stochastic = self.stochastic_sampling
        if not stochastic:
            return mean, v

        if t_next is None:
            t_next = (self._to_tau(t, xt) - (1.0 / float(self.timesteps))).clamp(min=0.0)
        else:
            t_next = self._to_tau(t_next, xt)
        should_sample = t_next > 0.0
        while should_sample.ndim < xt.ndim:
            should_sample = should_sample.unsqueeze(-1)
        noise = torch.randn_like(mean)
        x_next = torch.where(should_sample, mean + std * noise, mean)
        return x_next, v

    @torch.no_grad()
    def p_distribution(self, xt, y, t, t_next=None, indices=None, guid_params=None, return_output=False, use_ema: bool = True):
        if t.dim() == 1:
            t = t.unsqueeze(-1).expand(t.shape[0], xt.shape[1])
        t = self._to_tau(t, xt)
        if t_next is None:
            t_next = (t - (1.0 / float(self.timesteps))).clamp(min=0.0)
        else:
            t_next = self._to_tau(t_next, xt)
        v, logvar = self._guided_model_output(xt, y, t, use_ema=use_ema)
        sigma_theta = (
            torch.exp(0.5 * torch.clamp(logvar, min=-30.0, max=20.0))
            if logvar is not None
            else None
        )
        temperature = float(y["infos"].get("sampling_temperature", self.sampling_temperature))
        alpha = float(y["infos"].get("variance_alpha", self.variance_alpha))
        mean, sigma = self._transition_stats(
            xt=xt,
            v_theta=v,
            sigma_theta=sigma_theta,
            t=t,
            t_next=t_next,
            alpha=alpha,
            temperature=temperature,
        )
        if return_output:
            return mean, sigma, v
        return mean, sigma

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def diffusion_step(self, batch, batch_idx, training=False):
        mask = batch["mask"]
        x = masked(self.motion_normalizer(batch["x"]), mask)
        y = {
            "length": batch["length"],
            "mask": mask,
            "tx": self.prepare_tx_emb(batch["tx"]),
        }

        bs = len(x)
        self._init_time_sampler(x.shape)

        t_continuous, loss_weight = self.time_sampler(bs, num_samples=1, device=x.device)
        t_continuous = t_continuous.clamp(0.0, 1.0)
        if self.per_feature:
            t_per_frame = t_continuous.squeeze(1).to(torch.float32).mean(dim=-1)
        else:
            t_per_frame = t_continuous.squeeze(1).to(torch.float32)

        if self.shared_timestep_per_frame:
            t_per_frame = t_per_frame[:, :1].expand_as(t_per_frame)
            if loss_weight.ndim == 3 and loss_weight.shape[-1] == t_per_frame.shape[1]:
                loss_weight = loss_weight[..., :1].expand_as(loss_weight)
            elif loss_weight.ndim == 4 and loss_weight.shape[-2] == t_per_frame.shape[1]:
                loss_weight = loss_weight[..., :1, :].expand_as(loss_weight)

        loss_weight_expanded = loss_weight.transpose(1, 2).expand_as(x)

        # ---- Forward diffusion + denoising (shared) -------------------
        noise = masked(torch.randn_like(x), mask)
        t_expanded = t_per_frame.unsqueeze(-1).expand_as(x)
        xt = masked(self.flow_forward_sample(x, t_expanded, noise=noise), mask)

        out = self.denoiser(xt, y, t_per_frame)
        if isinstance(out, tuple):
            v_pred, output_logvar = out
        else:
            v_pred, output_logvar = out, None
        v_pred = masked(v_pred, mask)
        target_v = masked(noise - x, mask)

        vloss_unweighted = self.reconstruction_loss(v_pred, target_v)
        vloss_weighted = vloss_unweighted * loss_weight_expanded
        loss_total = vloss_weighted.mean()
        sigma_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # ---- Sigma loss -----------------------------------------------
        if self.predict_sigma and output_logvar is not None and self.sigma_weight > 0.0:
            output_logvar = masked(output_logvar, mask)
            output_logvar = torch.clamp(output_logvar, min=-30.0, max=20.0)

            sigma_sample_mask = None
            sigma_target = target_v
            sigma_pred = v_pred.detach()
            sigma_nll = 0.5 * (
                log(2.0 * torch.pi)
                + (sigma_target - sigma_pred) ** 2 * torch.exp(-output_logvar)
                + output_logvar
            )
            sigma_nll_weighted = sigma_nll * loss_weight_expanded

            if sigma_sample_mask is not None:
                # Zero-out sigma loss for non-BABEL samples, normalise by
                # number of BABEL samples to keep gradient magnitude stable.
                sigma_nll_weighted = sigma_nll_weighted * sigma_sample_mask[:, None, None]
                n_babel = sigma_sample_mask.sum().clamp(min=1.0)
                sigma_loss = sigma_nll_weighted.sum() / (n_babel * sigma_nll_weighted.shape[1] * sigma_nll_weighted.shape[2])
            else:
                sigma_loss = sigma_nll_weighted.mean()

            loss_total = loss_total + self.sigma_weight * sigma_loss

        if vloss_weighted.size(-1) == 32:
            recon_motion16 = vloss_weighted[..., :16].mean()
            recon_text16 = vloss_weighted[..., -16:].mean()
        else:
            recon_motion16 = loss_total
            recon_text16 = loss_total

        return {
            "loss": loss_total,
            "recon": loss_total,
            "recon_unweighted": vloss_unweighted.mean(),
            "recon_motion16": recon_motion16,
            "recon_text16": recon_text16,
            "sigma": sigma_loss,
        }

    def training_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx, training=True)
        for loss_name in sorted(loss):
            self.log(
                f"train_{loss_name}",
                loss[loss_name],
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
        return loss["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema and self.ema_denoiser is not None:
            self.ema_denoiser.update_parameters(self.denoiser)

    def validation_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx, training=False)
        for loss_name in sorted(loss):
            self.log(
                f"val_{loss_name}",
                loss[loss_name],
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
        return loss["loss"]

    def on_validation_epoch_end(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        is_latent = infer_is_latent_model(self)
       # run_validation_eval(trainer, self, is_latent=is_latent)

    def on_train_epoch_end(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        self._saved_losses = defaultdict(list)
        self.losses = []
        self.log_dict(
            {
                "epoch": float(trainer.current_epoch),
                "step": float(trainer.global_step),
            }
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["diffusion_type"] = self.name

    def forward(
        self,
        tx_emb,
        tx_emb_uncond,
        infos,
        progress_bar=tqdm,
        gt_motion=None,
        keyframe_indices=None,
        use_ema: bool = True,
    ):
        return self.text_forward(
            tx_emb,
            tx_emb_uncond,
            infos,
            progress_bar=progress_bar,
            gt_motion=gt_motion,
            keyframe_indices=keyframe_indices,
            use_ema=use_ema,
        )

    def text_forward(
        self,
        tx_emb,
        tx_emb_uncond,
        infos,
        progress_bar=tqdm,
        keyframe_indices=None,
        gt_motion=None,
        use_ema: bool = True,
    ):
        device = self.device
        lengths = infos["all_lengths"]
        mask = length_to_mask(lengths, device=device)
        bs = len(lengths)
        duration = self.inference_length if self.inference_length is not None else max(lengths)
        nfeats = self.denoiser.nfeats

        if mask.shape[1] < duration:
            pad_len = duration - mask.shape[1]
            pad = torch.zeros(mask.shape[0], pad_len, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([mask, pad], dim=1)
        elif mask.shape[1] > duration:
            mask = mask[:, :duration]

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }
        stochastic = bool(infos.get("stochastic_sampling", self.stochastic_sampling))

        xt = torch.randn((bs, duration, nfeats), device=device)
        tau_grid = torch.linspace(1.0, 0.0, self.timesteps + 1, device=device, dtype=xt.dtype)
        iterator = list(zip(tau_grid[:-1], tau_grid[1:]))
        if progress_bar is not None:
            iterator = progress_bar(iterator, desc="RectifiedFlow")

        for tau, tau_next in iterator:
            t = torch.full((bs, duration), float(tau), device=device, dtype=xt.dtype)
            t_next = torch.full((bs, duration), float(tau_next), device=device, dtype=xt.dtype)
            xt, _ = self.p_sample(xt, y, t, t_next=t_next, stochastic=stochastic, use_ema=use_ema)

        xstart = self.motion_normalizer.inverse(masked(xt, mask))
        return xstart
