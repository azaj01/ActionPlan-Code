"""ActionPlan transformer denoiser for two-stream rectified flow.

Extends TransformerDenoiserContinuousT with a second ContinuousTimestepEmbedder
so the model can receive independent noise-level signals for the motion stream
(dims 0:16) and the text stream (dims 16:32).

``t`` must be 3-D ``[bs, nframes, 2]``:
    t[..., 0] → motion timestep
    t[..., 1] → text timestep
"""

import torch
import torch.nn as nn

from .denoiser import TransformerDenoiserContinuousT
from .positional_encoding import ContinuousTimestepEmbedder
from einops import repeat


class TransformerDenoiserActionPlanT(TransformerDenoiserContinuousT):
    """Transformer denoiser with two independent continuous-timestep embedders."""

    name = "transformer_actionplan_t"
    supports_continuous_t = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The parent already created self.timestep_encoder (motion stream).
        # Add a second embedder for the text stream.
        self.timestep_encoder_text = ContinuousTimestepEmbedder(self.latent_dim)

    def forward(self, x, y, t):
        device = x.device
        x_mask = y["mask"]
        bs, nframes, nfeats = x.shape

        # ---- Split actionplan timestep -----------------------------------------
        if t.ndim != 3 or t.shape[-1] != 2:
            raise ValueError(
                f"TransformerDenoiserActionPlanT expects t of shape [bs, nframes, 2], "
                f"got {list(t.shape)}. Use the actionplan diffusion pipeline."
            )
        t_motion = t[..., 0]  # [bs, nframes]
        t_text = t[..., 1]    # [bs, nframes]

        # ---- Time embeddings (sum of two streams) -----------------------
        time_emb = self.timestep_encoder(t_motion) + self.timestep_encoder_text(t_text)

        # ---- Condition tokens -------------------------------------------
        assert "tx" in y
        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"]
        tx_emb = self.tx_embedding(tx_x)

        info_emb = tx_emb
        info_mask = tx_mask

        if self.nb_registers > 0:
            registers = repeat(self.registers, "nbtoken dim -> bs nbtoken dim", bs=bs)
            registers_mask = torch.ones(
                (bs, self.nb_registers), dtype=bool, device=device
            )
            info_emb = torch.cat((info_emb, registers), 1)
            info_mask = torch.cat((info_mask, registers_mask), 1)

        # ---- Embed motion features and add time -------------------------
        x = self.skel_embedding(x)
        number_of_info = info_emb.shape[1]
        x = x + time_emb

        # ---- Transformer ------------------------------------------------
        xseq = torch.cat((info_emb, x), 1)
        xseq = self.sequence_pos_encoding(xseq)
        aug_mask = torch.cat((info_mask, x_mask), 1)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # ---- Output heads ------------------------------------------------
        h = final[:, number_of_info:]
        mean_pred = self.to_skel_layer(h)
        if self.predict_sigma and self.to_logvar_layer is not None:
            logvar_pred = self.to_logvar_layer(h)
            return mean_pred, logvar_pred
        return mean_pred
