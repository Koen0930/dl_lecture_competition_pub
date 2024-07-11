import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import models

from .model import MEGEncoder, Enc_eeg
from typing import List


class CLIPModel(nn.Module):
    def __init__(
        self,
        position_list: List[int],
        im_weight_path: str,
        image_encoder: str,
        meg_encoder: str,
        num_classes: int
    ) -> None:
        super().__init__()
        if image_encoder == "resnet50":
            self.emb_dim = 2048
            self.ImageEncoder = models.resnet50()
            self.ImageEncoder.fc = nn.Linear(self.ImageEncoder.fc.in_features, num_classes)
            self.ImageEncoder.load_state_dict(torch.load(im_weight_path))
            self.ImageEncoder.fc = nn.Identity()
        elif image_encoder == "efficientnet_v2_s":
            self.emb_dim = 1280
            self.ImageEncoder = models.efficientnet_v2_s()
            self.ImageEncoder.classifier[1] = nn.Linear(self.ImageEncoder.classifier[1].in_features, num_classes)
            self.ImageEncoder.load_state_dict(torch.load(im_weight_path))
            self.final_layer = self.ImageEncoder.classifier[1]
            self.ImageEncoder.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported image encoder: {image_encoder}")
        
        if meg_encoder == "eeg":
            self.meg_encoder = Enc_eeg(position_list, out_channels=self.emb_dim)
            
        elif meg_encoder == "meg":
            self.meg_encoder = MEGEncoder(position_list, out_channels=self.emb_dim)
        for param in self.ImageEncoder.parameters():
            param.requires_grad = False
        # for param in self.final_layer.parameters():
        #     param.requires_grad = False

    def forward(self, image: torch.Tensor, meg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        encoded_image = self.ImageEncoder(image)
        encoded_meg = self.meg_encoder(meg, subject)
        return encoded_image, encoded_meg


class CLIPLoss(nn.Module):
    def __init__(self, temperature=1) -> None:
        super().__init__()
        self.temperature = torch.tensor(temperature)

    def create_label_matrix(self, y: torch.Tensor) -> torch.Tensor:
        # Create a zero matrix of shape (batch_size, batch_size)
        label_matrix = torch.eye(y.size(0), device=y.device)
        
        # Set the appropriate elements to 1
        y = y.unsqueeze(0)
        label_matrix = (y == y.T).float()
        return label_matrix
    
    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Vectorized cross entropy loss
        loss = -(labels * torch_log(logits) + (1 - labels) * torch_log(1 - logits))
        return loss.sum()

    def forward(self, encoded_image: torch.Tensor, encoded_meg: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Normalize the embeddings
        encoded_image = F.normalize(encoded_image, dim=-1)
        encoded_meg = F.normalize(encoded_meg, dim=-1)
        
        logits = torch.matmul(encoded_image, encoded_meg.T) * torch.exp(self.temperature)

        # symmetric loss function
        labels = torch.arange(encoded_image.size(0))
        labels = labels.to(logits.device)
        loss_i = F.cross_entropy(logits.transpose(0, 1), labels)
        loss_t = F.cross_entropy(logits, labels)
        loss = (loss_i + loss_t)/2
        # labels = self.create_label_matrix(y)
        # loss = self.cross_entropy_loss(logits, labels)
        
        return loss

def torch_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-10))


class ClassifierModel(nn.Module):
    def __init__(self, clip_model: CLIPModel) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.meg_encoder = clip_model.meg_encoder
        self.classifier = clip_model.final_layer
        self.classifier.out_features = 1854
        
    def forward(self, X: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        encoded_meg = self.meg_encoder(X, subject)
        return self.classifier(encoded_meg)