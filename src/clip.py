import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torchvision import models

from .model import MEGEncoder, Enc_eeg
from typing import List


def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                

class CLIPModel(nn.Module):
    def __init__(
        self,
        position_list: List[int],
        im_weight_path: str,
        image_encoder: str,
        meg_encoder: str,
        num_classes: int,
        gat_dropout=0.05681150010578299,
        graph_k=154,
        pa_dropout_rate=0.11003528220351602,
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
            self.meg_encoder = Enc_eeg(
                position_list=position_list,
                gat_dropout=gat_dropout,
                graph_k=graph_k,
                pa_dropout_rate=pa_dropout_rate,
                out_channels=self.emb_dim
            )
            
        elif meg_encoder == "meg":
            self.meg_encoder = MEGEncoder(position_list, out_channels=self.emb_dim)
        for param in self.ImageEncoder.parameters():
            param.requires_grad = False
        # for param in self.final_layer.parameters():
        #     param.requires_grad = False
        self.meg_encoder.apply(initialize_weights)

    def forward(self, image: torch.Tensor, meg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        encoded_image = self.ImageEncoder(image)
        encoded_meg = self.meg_encoder(meg, subject)
        return encoded_image, encoded_meg


class CLIPLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, encoded_image: torch.Tensor, encoded_meg: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Normalize the embeddings
        encoded_image = F.normalize(encoded_image, dim=-1)
        encoded_meg = F.normalize(encoded_meg, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        # image_cos_similarity_matrix = torch.mm(encoded_image, encoded_image.transpose(0, 1))
        # meg_cos_similarity_matrix = torch.mm(encoded_meg, encoded_meg.transpose(0, 1))
        # meg_img_cos_similarity = F.cosine_similarity(meg_cos_similarity_matrix, image_cos_similarity_matrix)
        # meg_img_cos_sim_loss = 1 - meg_img_cos_similarity.mean()
        
        logits_per_image = logit_scale * encoded_image @ encoded_meg.t()
        logits_per_meg = logits_per_image.t()
        # logits = torch.matmul(logits_per_image, encoded_meg.T) * torch.exp(self.temperature)

        # symmetric loss function
        labels = torch.arange(encoded_image.size(0))
        labels = labels.to(logits_per_image.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_m = F.cross_entropy(logits_per_meg, labels)
        loss = (loss_i + loss_m)/2
        # labels = self.create_label_matrix(y)
        # loss = self.cross_entropy_loss(logits, labels)
        
        return loss


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