import random
import numpy as np
import torch
import math

from torch import nn
import torch.nn.functional as F

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FourierEmb(nn.Module):
    """
    Fourier positional embedding.
    Unlike trad. embedding this is not using exponential periods
    for cosines and sinuses, but typical `2 pi k` which can represent
    any function over [0, 1]. As this function would be necessarily periodic,
    we take a bit of margin and do over [-0.2, 1.2].
    """
    def __init__(self, dimension: int = 256, margin: float = 0.2):
        super().__init__()
        n_freqs = (dimension // 2)**0.5
        assert int(n_freqs ** 2 * 2) == dimension
        self.dimension = dimension
        self.margin = margin

    def forward(self, positions):
        *O, D = positions.shape
        assert D == 2
        *O, D = positions.shape
        n_freqs = (self.dimension // 2)**0.5
        freqs_y = torch.arange(n_freqs).to(positions)
        freqs_x = freqs_y[:, None]
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        positions = positions[..., None, None, :]
        loc = (positions[..., 0] * p_x + positions[..., 1] * p_y).view(*O, -1)
        emb = torch.cat([
            torch.cos(loc),
            torch.sin(loc),
        ], dim=-1)
        return emb


class ChannelMerger(nn.Module):
    def __init__(self, chout: int, positions: torch.Tensor, pos_dim: int = 256,
                 dropout: float = 0, usage_penalty: float = 0.,
                 n_subjects: int = 4):
        super().__init__()

        self.positions = positions
        self.heads = nn.Parameter(torch.randn(n_subjects, chout, pos_dim, requires_grad=True))
        self.heads.data /= pos_dim ** 0.5
        self.dropout = dropout
        self.embedding = FourierEmb(pos_dim)
        self.usage_penalty = usage_penalty
        self._penalty = torch.tensor(0.)

    @property
    def training_penalty(self):
        return self._penalty.to(next(self.parameters()).device)

    def forward(self, meg, subject):
        B, C, T = meg.shape
        meg = meg.clone()
        # self.positionsをB繰り返す
        positions = self.positions.expand(B, -1, -1)
        embedding = self.embedding(positions)
        score_offset = torch.zeros(B, C, device=meg.device)

        if self.training and self.dropout:
            center_to_ban = torch.rand(2, device=meg.device)
            radius_to_ban = self.dropout
            banned = (positions - center_to_ban).norm(dim=-1) <= radius_to_ban
            score_offset[banned] = float('-inf')

        _, cout, pos_dim = self.heads.shape
        heads = self.heads.gather(0, subject.view(-1, 1, 1).expand(-1, cout, pos_dim))
        scores = torch.einsum("bcd,bod->boc", embedding.float(), heads)
        scores += score_offset[:, None]
        weights = torch.softmax(scores, dim=2)
        out = torch.einsum("bct,boc->bot", meg, weights)
        if self.training and self.usage_penalty > 0.:
            usage = weights.mean(dim=(0, 1)).sum()
            self._penalty = self.usage_penalty * usage
        return out


class InitialLayer(nn.Module):
    def __init__(self, initial_linear: int=271, initial_depth: int=1):
        super().__init__()
        self.initial_linear = initial_linear
        self.initial_depth = initial_depth
        self.activation = nn.GELU
        init = [nn.Conv1d(271, self.initial_linear, 1)]
        for _ in range(self.initial_depth - 1):
            init += [self.activation(), nn.Conv1d(self.initial_linear, self.initial_linear, 1)]
        self.initial_layer = nn.Sequential(*init)

    def forward(self, x):
        return self.initial_layer(x)


class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int = 4, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

    def __repr__(self):
        S, C, D = self.weights.shape
        return f"SubjectLayers({C}, {D}, {S})"


class ResidualDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, linear_dim=2048, dilation_base=2, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        D2 = 320
        for k in range(num_blocks):
            dilation1 = dilation_base ** ((2 * k) % 5)
            dilation2 = dilation_base ** ((2 * k + 1) % 5)
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_channels if k == 0 else D2, D2, kernel_size=3, padding=dilation1, dilation=dilation1),
                # nn.Conv1d(in_channels if k == 0 else D2, D2, kernel_size=3, dilation=dilation1),
                nn.BatchNorm1d(D2),
                nn.GELU(),
                nn.Conv1d(D2, D2, kernel_size=3, padding=dilation2, dilation=dilation2),
                # nn.Conv1d(D2, D2, kernel_size=3, dilation=dilation2),
                nn.BatchNorm1d(D2),
                nn.GELU(),
                nn.Conv1d(D2, 2 * D2, kernel_size=3, padding=1),
                nn.GLU(dim=1)
            ))
        self.final_conv1 = nn.Conv1d(D2, linear_dim, kernel_size=1)
        self.affine_projection = nn.Conv1d(281, 1, kernel_size=1)  # Affine projection layer
        self.final_linear = nn.Linear(linear_dim, out_channels)
        # self.final_gelu = nn.GELU()
        # self.final_conv2 = nn.Conv1d(linear_dim, out_channels, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            residual = x
            x = block(x)
            if residual.shape[1] == x.shape[1]:
                x += residual
        x = self.final_conv1(x)
        x = x.permute(0, 2, 1)
        x = self.affine_projection(x)
        # 時間次元の削除
        x = x.squeeze(dim=1)
        # x = self.final_gelu(x)
        # x = self.final_conv2(x)
        x = self.final_linear(x)
        return x


class MEGEncoder(nn.Module):
    def __init__(
        self,
        positions,
        pos_dim: int = 32,
        merger_dropout: float = 0.2,
        usage_penalty: float = 0.,
        n_subjects: int = 4,
        in_channels: int = 271,
        out_channels: int = 2048,
        init_id: bool = False,
    ):
        super().__init__()
        self.channel_merger = ChannelMerger(
            chout=in_channels,
            positions=positions,
            pos_dim=pos_dim,
            dropout=merger_dropout,
            usage_penalty=usage_penalty,
            n_subjects=n_subjects
        )
        self.initial_layer = InitialLayer(initial_linear=in_channels, initial_depth=1)
        self.subject_layers = SubjectLayers(in_channels, in_channels, n_subjects, init_id)
        self.residual_dilated_conv_block = ResidualDilatedConvBlock(in_channels, out_channels)

    def forward(self, x, subjects):
        x = self.channel_merger(x, subjects)
        x = self.initial_layer(x)
        x = self.subject_layers(x, subjects)
        x = self.residual_dilated_conv_block(x)
        return x
