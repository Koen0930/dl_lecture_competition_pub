import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import models

from .utils import MEGEncoder


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

class CLIPModel(nn.Module):
    def __init__(
        self,
        postion_list: List[int],
        im_weight_path: str,
        image_encoder: str,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        if image_encoder == "resnet50":
            self.ImageEncoder = models.resnet50(weights=im_weight_path)
            self.meg_encoder = MEGEncoder(position_list, out_channels=2048)
        elif image_encoder == "efficientnet_v2_s":
            self.ImageEncoder = models.efficientnet_v2_s(weights=im_weight_path)
            self.meg_encoder = MEGEncoder(position_list, out_channels=1280)
        else:
            raise ValueError(f"Unsupported image encoder: {image_encoder}")

    def forward(self, image: torch.Tensor, meg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        encoded_image = self.ImageEncoder(image)
        encoded_meg = self.meg_encoder(meg, subject)
        return encoded_image, encoded_meg


class CLIPLoss(nn.Module):
    def __init__(self, temperature=1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, encoded_image: torch.Tensor, encoded_meg: torch.Tensor) -> torch.Tensor:
        # Normalize the embeddings
        encoded_image = F.normalize(encoded_image, dim=-1)
        encoded_meg = F.normalize(encoded_meg, dim=-1)
        
        logits = torch.matmul(encoded_image, encoded_meg.T) * torch.exp(self.temperature)
        loss = F.cross_entropy(logits, labels)

        # symmetric loss function
        labels = np.arange(n)
        loss_i = cross_entropy_loss(logits, labels, axis=0)
        loss_t = cross_entropy_loss(logits, labels, axis=1)
        loss = (loss_i + loss_t)/2
        
        return loss