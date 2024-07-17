import os, sys
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from torchmetrics import Accuracy
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from topk.svm import SmoothTopkSVM

from src.datasets import ImageDataset
from src.models import BasicConvClassifier, ConvRNNClassifier, BasicLSTMClassifier
from src.utils import set_seed
from src.conformer import Conformer
from src.utils import MEGEncoder

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = 0'  # すべてのGPUを指定

@hydra.main(version_base=None, config_path="configs", config_name="image_config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="Image-classification")
    
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
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ImageDataset("train", args.data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ImageDataset("val", args.data_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    
    # ------------------
    #       Model
    # ------------------
    # 学習済みモデルの読み込み
    # Resnet50を重み付きで読み込む
    if args.model == "resnet50":
        model_ft = models.resnet50(pretrained=True)
        # 最終ノードの出力を変更する
        model_ft.fc = nn.Linear(model_ft.fc.in_features, train_set.num_classes)
        net = model_ft.to(args.device)
    elif args.model == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        # 最終ノードの出力を変更する
        model_ft.fc = nn.Linear(model_ft.fc.in_features, train_set.num_classes)
        net = model_ft.to(args.device)
    elif args.model == "efficientnet_v2_s":
        model_ft = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        # 最終ノードの出力を変更する
        model_ft.classifier[1] = nn.Linear(model_ft.classifier[1].in_features, train_set.num_classes)
        net = model_ft.to(args.device)

    
    
    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes
    ).to(args.device)
    
    loss_fn = F.cross_entropy
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        net.train()
        for X, y in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)
            y_pred = net(X)
            
            loss = loss_fn(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        net.eval()
        for X, y in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            with torch.no_grad():
                y_pred = net(X)
            
            val_loss.append(loss_fn(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(net.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(net.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)


if __name__ == "__main__":
    run()