import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding, TimestepEmbedder, ContinuousTimestepEmbedder
from einops import repeat


class TransformerDenoiser(nn.Module):
    name = "transformer"

    def __init__(
        self,
        nfeats: int,
        tx_dim: int,
        latent_dim: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        nb_registers: int = 2,
        activation: str = "gelu",
        predict_sigma: bool = False,
    ):
        super().__init__()

        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.nb_registers = nb_registers
        self.tx_dim = tx_dim

        # Linear layer for the condition
        self.tx_embedding = nn.Sequential(
            nn.Linear(tx_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Linear layer for the skeletons
        self.skel_embedding = nn.Linear(nfeats, latent_dim)

        # register for aggregating info
        if nb_registers > 0:
            self.registers = nn.Parameter(torch.randn(nb_registers, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        # MLP for the timesteps
        self.timestep_encoder = TimestepEmbedder(latent_dim, self.sequence_pos_encoding)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            norm_first=True,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        # Final layer to go back to skeletons
        self.to_skel_layer = nn.Linear(latent_dim, nfeats)
        # Optional log-variance head for heteroscedastic sigma (per feature per frame)
        self.predict_sigma = predict_sigma
        if self.predict_sigma:
            self.to_logvar_layer = nn.Linear(latent_dim, nfeats)
            # Initialize logvar layer to output near-zero values for stable training start
            # This corresponds to log(1) = 0, i.e., unit variance initially
            nn.init.zeros_(self.to_logvar_layer.weight)
            nn.init.zeros_(self.to_logvar_layer.bias)
        else:
            self.to_logvar_layer = None

    def forward(self, x, y, t):
        device = x.device
        x_mask = y["mask"]
        bs, nframes, nfeats = x.shape

        # Time embedding
        time_emb = self.timestep_encoder(t)


        assert "tx" in y

        # Condition part (can be text/action etc)
        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"]

        tx_emb = self.tx_embedding(tx_x)

        info_emb =  tx_emb
        info_mask =  tx_mask
        # add registers
        if self.nb_registers > 0:
            registers = repeat(self.registers, "nbtoken dim -> bs nbtoken dim", bs=bs)
            registers_mask = torch.ones(
                (bs, self.nb_registers), dtype=bool, device=device
            )
            # add the register
            info_emb = torch.cat((info_emb, registers), 1)
            info_mask = torch.cat((info_mask, registers_mask), 1)

        x = self.skel_embedding(x)
        number_of_info = info_emb.shape[1]

        # Add time embedding directly to the sequence embeddings (per-frame conditioning)
        x = x + time_emb
        
        # adding the embedding token for all sequences
        xseq = torch.cat((info_emb, x), 1)

        # add positional encoding to all the tokens
        xseq = self.sequence_pos_encoding(xseq)

        # create a bigger mask, to allow attend to time and condition as well
        aug_mask = torch.cat((info_mask, x_mask), 1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # extract the important part
        h = final[:, number_of_info:]
        mean_pred = self.to_skel_layer(h)
        if self.predict_sigma and self.to_logvar_layer is not None:
            logvar_pred = self.to_logvar_layer(h)
            return mean_pred, logvar_pred
        return mean_pred


class TransformerDenoiserContinuousT(TransformerDenoiser):
    """Same architecture as TransformerDenoiser with continuous-time embedding."""

    name = "transformer_continuous_t"
    supports_continuous_t = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestep_encoder = ContinuousTimestepEmbedder(self.latent_dim)
