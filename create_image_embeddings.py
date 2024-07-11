import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from torchvision import transforms
from torchvision import models
from torchvision import utils
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

from src.utils import set_seed
from src.datasets import ImageDataset


def run():
    set_seed(42)
    save_dir = "data"
    device = "cuda:0"
    # ------------------
    #       Transform
    # ------------------
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # ------------------
    #    Dataloader
    # ------------------
    print("* data loading")
    loader_args = {"batch_size": 1, "num_workers": 8, "pin_memory": False, "shuffle": False}
    train_set = ImageDataset("train", "data", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **loader_args)
    val_set = ImageDataset("val", "data", transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, **loader_args)
    print("* data loading done")
    # ------------------
    #       Model
    # ------------------
    im_weight_path = "/root/outputs/2024-07-08/20-51-06/model_best.pt"
    model = models.efficientnet_v2_s()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, train_set.num_classes)
    model.load_state_dict(torch.load(im_weight_path))
    model.classifier = nn.Identity()
    model.eval()
    model.to(device)
    print("* model loading done")
    # ------------------
    #       Embedding
    # ------------------
    # Embeddingsリストを初期化
    embeddings_list = []

    # ラベルごとにデータを分類
    label_to_data = {}
    for X, y in tqdm(train_loader):
        label = y.item()
        if label not in label_to_data:
            label_to_data[label] = []
        label_to_data[label].append(X)
    print(len(label_to_data))

    # 各ラベルごとに処理
    for label in range(1854):
        X = torch.stack(label_to_data[label])
        embeddings = model(X.to(device))
        # 各行の和を計算
        embeddings = embeddings.sum(dim=0).to("cpu")
        embeddings_list.append(embeddings)
        
        
    embeddings_list = torch.cat(embeddings_list, dim=0)
    torch.save(embeddings_list, os.path.join(save_dir, "train_image_embeddings.pt"))
    
if __name__ == "__main__":
    run()