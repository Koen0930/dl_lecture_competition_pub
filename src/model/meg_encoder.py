import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange
from torchvision import models

from typing import List

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)

class ConvRNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim1: int = 256,
        hid_dim2: int = 128,
        hid_dim3: int = 64,
        rnn_hidden_dim: int = 1024,
        mlp_hidden_dim1: int = 2048,
        num_layers: int = 2,
        p_drop: float = 0.5
    ) -> None:
        super().__init__()

        self.cnn_blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim1, p_drop=p_drop),
            ConvBlock(hid_dim1, hid_dim2, p_drop=p_drop),
            nn.MaxPool1d(2),  # Added pooling layer
            ConvBlock(hid_dim2, hid_dim3, p_drop=p_drop),  # Added another ConvBlock
        )

        self.lstm = nn.LSTM(
            input_size=hid_dim3,  # Adjusted input size
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop
        )

        self.head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, mlp_hidden_dim1),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim1, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.cnn_blocks(X)
        X = X.permute(0, 2, 1)  # (b, c, t) -> (b, t, c) for RNN
        X, _ = self.lstm(X)
        X = X[:, -1, :]  # Take the last output of the RNN

        return self.head(X)
    

class BasicLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        rnn_hidden_dim: int = 256,
        num_layers: int = 3,
        p_drop: float = 0.7
    ) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=in_channels,  # Adjusted input size
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop
        )

        self.head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, num_classes),
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.permute(0, 2, 1)
        X, _ = self.lstm(X)
        X = X[:, -1, :]  # Take the last output of the RNN

        return self.head(X)

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


class PatchEmbedding(nn.Module):
    def __init__(self, out_channels=2048):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 20, (1, 15), (1, 1)),
            nn.AvgPool2d((1, 31), (1, 5)),
            nn.BatchNorm2d(20),
            nn.ELU(),
            nn.Conv2d(20, 10, (33, 1), (1, 1)),
            nn.BatchNorm2d(10),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(10, 1, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.linear = nn.Linear(9560, out_channels)

    def forward(self, x):
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        x = x.squeeze(-1)
        x = self.linear(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(
        self,
        positions,
        n_subjects: int = 4,
        in_channels: int = 271,
        out_channels: int = 2048,
        init_id: bool = False,
    ):
        super().__init__()
        self.initial_layer = InitialLayer(initial_linear=in_channels, initial_depth=1)
        self.subject_layers = SubjectLayers(in_channels, in_channels, n_subjects, init_id)
        self.patchembedding = PatchEmbedding(out_channels)
        
        self.ga = ResidualAdd(
            nn.Sequential(
                EEG_GAT(),
                nn.Dropout(0.3),
                )
        )
        self.ca = ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(244),
                    channel_attention(),
                    nn.Dropout(0.3),
                )
        )
        
    def forward(self, x, subjects):
        # x = self.ca(x)
        x = self.initial_layer(x)
        x = self.subject_layers(x, subjects)
        x = x.unsqueeze(dim=1)
        x = self.patchembedding(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, sequence_num=281, inter=1):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(sequence_num, sequence_num),
            nn.LayerNorm(sequence_num), 
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(sequence_num, sequence_num),
            nn.LayerNorm(sequence_num),
            nn.Dropout(0.3)
        )
        self.value = nn.Sequential(
            nn.Linear(sequence_num, sequence_num),
            nn.LayerNorm(sequence_num),
            nn.Dropout(0.3)
        )

        self.projection = nn.Sequential(
            nn.Linear(sequence_num, sequence_num),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        channel_query = self.query(x)
        channel_key = self.key(x)
        channel_value = self.value(x)

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = torch.bmm(channel_query, torch.transpose(channel_key, 2, 1)) / scaling

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.bmm(channel_atten_score, channel_value)

        out = self.projection(out)
        return out


from torch_geometric.nn import GATConv
class EEG_GAT(nn.Module):
    def __init__(self, in_channels=271, out_channels=271):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
        # self.conv2 = GATConv(in_channels=out_channels, out_channels=out_channels, heads=1)

        self.num_channels = 64
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).cuda()
        # Convert the list of tuples to a tensor
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().cuda()

    def forward(self, x):

        batch_size, num_channels, num_features = x.size()
        x = x.view(batch_size*num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        
        return x