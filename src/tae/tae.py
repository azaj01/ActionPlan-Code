# Adapted from MotionStreamer: models/tae.py
# Upstream repo: https://github.com/zju3dv/MotionStreamer/
# Source file: https://github.com/zju3dv/MotionStreamer/blob/main/models/tae.py

"""Temporal AutoEncoder (TAE) model.

The TAE encodes 272-dim motion sequences into 16-dim latent space with 4x temporal compression.
"""

import torch.nn as nn
from .causal_cnn import CausalEncoder, CausalDecoder


class Causal_TAE(nn.Module):
    """Causal Temporal AutoEncoder for motion sequences.
    
    Encodes 272-dim motion features to 16-dim latent space with 4x temporal downsampling.
    
    Args:
        hidden_size: Hidden dimension (default: 1024)
        down_t: Number of temporal downsampling layers (default: 2, giving 4x compression)
        stride_t: Stride for temporal convolutions (default: 2)
        width: Width of internal layers (default: 1024)
        depth: Depth of ResNet blocks (default: 3)
        dilation_growth_rate: Dilation growth rate (default: 3)
        activation: Activation function (default: 'relu')
        norm: Normalization type (default: None)
        latent_dim: Latent dimension (default: 16)
        clip_range: Range for clipping logvar (default: [-30, 20])
    """

    def __init__(
        self,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        width=1024,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        latent_dim=16,
        clip_range=[]
    ):
        super().__init__()

        self.decode_proj = nn.Linear(latent_dim, width)

        self.encoder = CausalEncoder(
            272, hidden_size, down_t, stride_t, width, depth,
            dilation_growth_rate, activation=activation, norm=norm,
            latent_dim=latent_dim, clip_range=clip_range
        )
        self.decoder = CausalDecoder(
            272, hidden_size, down_t, stride_t, width, depth,
            dilation_growth_rate, activation=activation, norm=norm
        )

    def preprocess(self, x):
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        x_in = self.preprocess(x)
        x_encoder, mu, logvar = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])

        return x_encoder, mu, logvar

    def forward(self, x):
        x_in = self.preprocess(x)
        # Encode
        x_encoder, mu, logvar = self.encoder(x_in)
        x_encoder = self.decode_proj(x_encoder)
        # Decoder
        x_decoder = self.decoder(x_encoder)
        x_out = self.postprocess(x_decoder)
        return x_out, mu, logvar

    def forward_decoder(self, x):
        """Decode latent representation to 272-dim motion.
        
        Args:
            x: Latent tensor of shape (batch, seq_len, 16) or (seq_len, 16)
        
        Returns:
            Decoded motion of shape (batch, seq_len*4, 272)
        """
        # decoder
        x_width = self.decode_proj(x)
        x_decoder = self.decoder(x_width)
        x_out = self.postprocess(x_decoder)
        return x_out


class Causal_HumanTAE(nn.Module):
    """Wrapper class for Causal TAE with human motion specific defaults.
    
    This is the main class to use for loading checkpoints.
    """

    def __init__(
        self,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        latent_dim=16,
        clip_range=[]
    ):
        super().__init__()
        self.tae = Causal_TAE(
            hidden_size, down_t, stride_t, hidden_size, depth,
            dilation_growth_rate, activation=activation, norm=norm,
            latent_dim=latent_dim, clip_range=clip_range
        )

    def encode(self, x):
        h, mu, logvar = self.tae.encode(x)
        return h, mu, logvar

    def forward(self, x):
        x_out, mu, logvar = self.tae(x)
        return x_out, mu, logvar

    def forward_decoder(self, x):
        """Decode latent representation to 272-dim motion.
        
        Args:
            x: Latent tensor of shape (batch, seq_len, 16) or (seq_len, 16)
        
        Returns:
            Decoded motion of shape (batch, seq_len*4, 272)
        """
        if(x.shape[2] == 32):
            x = x[:, :, :16] ## This is a bit hacky to support t2m_latent_frame_text_aligned
        x_out = self.tae.forward_decoder(x)
        return x_out

