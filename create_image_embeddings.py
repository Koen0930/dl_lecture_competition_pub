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
    device = "cpu"
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
    loader_args = {"batch_size": 1, "num_workers": 4, "pin_memory": False, "shuffle": False}
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
    embeddings_list = []
    # for X, y in tqdm(train_loader, desc="Train"):
    #     X, y = X.to(device), y.to(device)
    #     embeddings = model(X)
    #     embeddings = embeddings.cpu()
    #     embeddings_list.append(embeddings)
    #     # 使い終わったXとyをGPUから外す
    #     del X, y
    #     torch.cuda.empty_cache()  # 未使用のメモリを解放
        
    for X, y in tqdm(val_loader, desc="Val"):
        X, y = X.to(device), y.to(device)
        embeddings = model(X)
        embeddings = embeddings.cpu()
        embeddings_list.append(embeddings)
    embeddings = torch.cat(embeddings_list, dim=0)
    torch.save(embeddings, os.path.join(save_dir, "image_embeddings.pt"))
    
if __name__ == "__main__":
    run()