# Adapted from STMC: src/model/positional_encoding.py
# Upstream repo: https://github.com/nv-tlabs/stmc
# Source file: https://github.com/nv-tlabs/stmc/blob/main/src/model/positional_encoding.py

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Adapted from MDM
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps][..., 0, :])


def continuous_timestep_encoding(timesteps, dim, max_period=10000):
    """Create sinusoidal embeddings for continuous timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.to(torch.float32).unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        t_emb = continuous_timestep_encoding(timesteps, self.latent_dim)
        return self.time_embed(t_emb)
