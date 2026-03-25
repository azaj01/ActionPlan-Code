"""ActionPlan cascaded rectified flow.

Two-stream architecture:
  - Motion stream: dims 0:16 (TAE motion latents)
  - Text stream:   dims 16:32 (AE-compressed frame-level CLIP embeddings)

Training: single text noise level and heterogeneous motion noise levels;
both streams denoise jointly. For samples without text latents (t2m_latents-only),
text timestep is 1.0 (pure noise) and loss is masked to motion-only.

Inference runs two cascaded phases:
  Phase 1: denoise text   (t_text 1→0,  t_motion = 1.0)
  Phase 2: denoise motion (t_motion 1→0, t_text   = 0.0)
"""

import logging
from math import log
from typing import Optional

import torch
from tqdm import tqdm

from .utils import masked
from .rectified_flow import RectifiedFlowDiffusion
from ..data.collate import length_to_mask

logger = logging.getLogger(__name__)

MOTION_DIM = 16
TEXT_DIM = 16


class ActionPlanRectifiedFlow(RectifiedFlowDiffusion):
    """Two-stream cascaded rectified flow with independent noise schedules."""

    name = "actionplan_rectified_flow"
    uses_rectified_flow = True

    def __init__(
        self,
        *args,
        text_steps: Optional[int] = None,
        motion_steps: Optional[int] = None,
        mode_a_ratio: Optional[float] = None,
        near_clean_max_t: Optional[float] = None,
        **kwargs,
    ):
        kwargs.pop("mode_a_ratio", None)
        kwargs.pop("near_clean_max_t", None)
        super().__init__(*args, **kwargs)
        self.text_steps = int(text_steps) if text_steps is not None else self.timesteps
        self.motion_steps = int(motion_steps) if motion_steps is not None else self.timesteps

    # ------------------------------------------------------------------
    # Guided model output (supports [bs, nframes, 2] timesteps)
    # ------------------------------------------------------------------

    def _guided_model_output_actionplan(self, xt, y, t, use_ema=True):
        """Classifier-free guidance with actionplan timestep ``t`` of shape ``[bs, nframes, 2]``."""
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

        bs, nframes, nfeats = x.shape
        device = x.device
        assert nfeats == MOTION_DIM + TEXT_DIM, (
            f"Expected {MOTION_DIM + TEXT_DIM} features, got {nfeats}"
        )

        has_text = batch.get("has_text_latent")
        if has_text is None:
            raise ValueError("has_text_latent is missing")
        else:
            has_text = has_text.to(device=device)

        self._init_time_sampler(x.shape)

        # ---- Motion: per-frame heterogeneous timesteps ------------------
        t_motion_sampled, motion_loss_weight = self.time_sampler(
            bs, num_samples=1, device=device,
        )
        t_motion = t_motion_sampled.clamp(0.0, 1.0).squeeze(1).to(torch.float32)  # [bs, nframes]
        # motion_loss_weight: [bs, 1, nframes] -> [bs, nframes, 1] -> expand to motion dims
        motion_lw = motion_loss_weight.transpose(1, 2).expand(bs, nframes, MOTION_DIM)
        # Text dims: uniform sampling -> weight = 1.0
        text_lw = torch.ones(bs, nframes, TEXT_DIM, device=device, dtype=x.dtype)
        time_loss_weight = torch.cat([motion_lw, text_lw], dim=-1)  # [bs, nframes, 32]

        # ---- Text: single noise level per sample U(0,1); t2m-only -> 1.0 ---
        t_text = torch.rand(bs, 1, device=device).expand(bs, nframes)
        t_text = torch.where(
            has_text.unsqueeze(1),
            t_text,
            torch.ones_like(t_text, device=device),
        )

        # ---- Loss mask: motion always; text only where has_text_latent ----
        loss_mask = torch.ones(bs, nframes, nfeats, device=device, dtype=x.dtype)
        loss_mask[~has_text, :, MOTION_DIM:] = 0.0

        # ---- Forward diffusion (both streams) ---------------------------
        noise = masked(torch.randn_like(x), mask)
        t_motion_exp = t_motion.unsqueeze(-1)
        t_text_exp = t_text.unsqueeze(-1)

        xt_motion = (
            (1.0 - t_motion_exp) * x[..., :MOTION_DIM]
            + t_motion_exp * noise[..., :MOTION_DIM]
        )
        xt_text = (
            (1.0 - t_text_exp) * x[..., MOTION_DIM:]
            + t_text_exp * noise[..., MOTION_DIM:]
        )
        xt = masked(torch.cat([xt_motion, xt_text], dim=-1), mask)

        # ---- Model forward (actionplan timesteps) ------------------------------
        t_model = torch.stack([t_motion, t_text], dim=-1)  # [bs, nframes, 2]
        out = self.denoiser(xt, y, t_model)
        if isinstance(out, tuple):
            v_pred, output_logvar = out
        else:
            v_pred, output_logvar = out, None
        v_pred = masked(v_pred, mask)
        target_v = masked(noise - x, mask)

        # ---- Velocity loss (masked by stream, weighted by time sampler) ----
        vloss_unweighted = self.reconstruction_loss(v_pred, target_v)
        vloss_weighted = vloss_unweighted * loss_mask * time_loss_weight
        loss_total = vloss_weighted.mean()

        sigma_loss = torch.tensor(0.0, device=device, dtype=x.dtype)
        if (
            self.predict_sigma
            and output_logvar is not None
            and self.sigma_weight > 0.0
        ):
            output_logvar = masked(output_logvar, mask)
            output_logvar = torch.clamp(output_logvar, min=-30.0, max=20.0)
            sigma_target = target_v
            sigma_pred = v_pred.detach()
            sigma_nll = 0.5 * (
                log(2.0 * torch.pi)
                + (sigma_target - sigma_pred) ** 2 * torch.exp(-output_logvar)
                + output_logvar
            )
            sigma_nll_weighted = sigma_nll * loss_mask * time_loss_weight
            sigma_loss = sigma_nll_weighted.mean()
            loss_total = loss_total + self.sigma_weight * sigma_loss

        with torch.no_grad():
            recon_motion16 = vloss_unweighted[..., :MOTION_DIM].mean()
            recon_text16 = vloss_unweighted[..., MOTION_DIM:].mean()

        return {
            "loss": loss_total,
            "recon": loss_total,
            "recon_unweighted": vloss_unweighted.mean(),
            "recon_motion16": recon_motion16,
            "recon_text16": recon_text16,
            "sigma": sigma_loss,
        }

    # ------------------------------------------------------------------
    # Inference: two-phase cascaded sampling
    # ------------------------------------------------------------------

    def _phase_step_stochastic(
        self,
        xt,
        v,
        logvar,
        dim_slice,
        t_val,
        t_next_val,
        alpha,
        temperature,
    ):
        """Single stochastic sampling step for one stream.

        Args:
            xt:          full noisy tensor [bs, dur, nfeats]
            v:           full velocity prediction [bs, dur, nfeats]
            logvar:      full log-variance prediction or None
            dim_slice:   ``slice(0, 16)`` or ``slice(16, 32)``
            t_val:       [bs, dur] current tau for the active stream
            t_next_val:  [bs, dur] next tau for the active stream
            alpha, temperature: sampler hyper-parameters

        Returns:
            updated ``xt`` with only the active stream modified.
        """
        v_stream = v[..., dim_slice]
        xt_stream = xt[..., dim_slice]

        sigma_theta = None
        if logvar is not None:
            sigma_theta = torch.exp(
                0.5 * torch.clamp(logvar[..., dim_slice], min=-30.0, max=20.0)
            )

        mean, std = self._transition_stats(
            xt=xt_stream,
            v_theta=v_stream,
            sigma_theta=sigma_theta,
            t=t_val,
            t_next=t_next_val,
            alpha=alpha,
            temperature=temperature,
        )

        # At the last step (t_next == 0) do not add noise
        should_sample = t_next_val > 0.0
        while should_sample.ndim < mean.ndim:
            should_sample = should_sample.unsqueeze(-1)

        noise = torch.randn_like(mean)
        stream_next = torch.where(should_sample, mean + std * noise, mean)

        # Reassemble full tensor
        parts = [xt[..., :dim_slice.start], stream_next, xt[..., dim_slice.stop:]]
        return torch.cat([p for p in parts if p.numel() > 0], dim=-1)

    @staticmethod
    def _phase_step_ode(xt, v, dim_slice, dt):
        """Single deterministic ODE step for one stream."""
        v_stream = v[..., dim_slice]
        stream_next = xt[..., dim_slice] - v_stream * dt
        parts = [xt[..., :dim_slice.start], stream_next, xt[..., dim_slice.stop:]]
        return torch.cat([p for p in parts if p.numel() > 0], dim=-1)

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
        duration = (
            self.inference_length if self.inference_length is not None else max(lengths)
        )
        nfeats = self.denoiser.nfeats

        if mask.shape[1] < duration:
            pad = torch.zeros(
                mask.shape[0], duration - mask.shape[1],
                dtype=mask.dtype, device=mask.device,
            )
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
        temperature = float(
            infos.get("sampling_temperature", self.sampling_temperature)
        )
        alpha = float(infos.get("variance_alpha", self.variance_alpha))

        # Start from pure noise
        xt = torch.randn((bs, duration, nfeats), device=device)

        text_slice = slice(MOTION_DIM, MOTION_DIM + TEXT_DIM)
        motion_slice = slice(0, MOTION_DIM)

        # ==============================================================
        # Phase 1 — denoise text stream (t_text: 1→0, t_motion stays 1)
        # ==============================================================
        tau_grid_text = torch.linspace(
            1.0, 0.0, self.text_steps + 1, device=device, dtype=xt.dtype,
        )
        text_iter = list(zip(tau_grid_text[:-1], tau_grid_text[1:]))
        if progress_bar is not None:
            text_iter = progress_bar(text_iter, desc="DualFlow Phase1 (text)")

        for tau, tau_next in text_iter:
            t_motion_val = torch.full(
                (bs, duration), 1.0, device=device, dtype=xt.dtype,
            )
            t_text_val = torch.full(
                (bs, duration), float(tau), device=device, dtype=xt.dtype,
            )
            t_text_next_val = torch.full(
                (bs, duration), float(tau_next), device=device, dtype=xt.dtype,
            )

            t_ap = torch.stack([t_motion_val, t_text_val], dim=-1)
            v, logvar = self._guided_model_output_actionplan(
                xt, y, t_ap, use_ema=use_ema,
            )

            if stochastic:
                xt = self._phase_step_stochastic(
                    xt, v, logvar, text_slice,
                    t_text_val, t_text_next_val, alpha, temperature,
                )
            else:
                dt = float(tau) - float(tau_next)
                xt = self._phase_step_ode(xt, v, text_slice, dt)

        # ==============================================================
        # Phase 2 — denoise motion stream (t_motion: 1→0, t_text stays 0)
        # ==============================================================
        tau_grid_motion = torch.linspace(
            1.0, 0.0, self.motion_steps + 1, device=device, dtype=xt.dtype,
        )
        motion_iter = list(zip(tau_grid_motion[:-1], tau_grid_motion[1:]))
        if progress_bar is not None:
            motion_iter = progress_bar(motion_iter, desc="DualFlow Phase2 (motion)")

        for tau, tau_next in motion_iter:
            t_motion_val = torch.full(
                (bs, duration), float(tau), device=device, dtype=xt.dtype,
            )
            t_text_val = torch.full(
                (bs, duration), 0.0, device=device, dtype=xt.dtype,
            )
            t_motion_next_val = torch.full(
                (bs, duration), float(tau_next), device=device, dtype=xt.dtype,
            )

            t_ap = torch.stack([t_motion_val, t_text_val], dim=-1)
            v, logvar = self._guided_model_output_actionplan(
                xt, y, t_ap, use_ema=use_ema,
            )

            if stochastic:
                xt = self._phase_step_stochastic(
                    xt, v, logvar, motion_slice,
                    t_motion_val, t_motion_next_val, alpha, temperature,
                )
            else:
                dt = float(tau) - float(tau_next)
                xt = self._phase_step_ode(xt, v, motion_slice, dt)

        xstart = self.motion_normalizer.inverse(masked(xt, mask))
        return xstart
