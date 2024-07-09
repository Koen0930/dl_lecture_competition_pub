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

from src.datasets import ImageMEGDataset
from src.models import CLIPModel, CLIPLoss
from src.utils import set_seed
from src.conformer import Conformer
from src.utils import MEGEncoder

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # すべてのGPUを指定

@hydra.main(version_base=None, config_path="configs", config_name="clip_config")
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
    
    print("* data loading")
    train_set = ImageMEGDataset("train", args.data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, pin_memory=True, **loader_args)
    val_set = ImageMEGDataset("val", args.data_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, pin_memory=True, **loader_args)
    
    position_list = torch.load("/root/data/position_list.pt").to(args.device)
    
    print("* data loading done")
    # ------------------
    #       Model
    # ------------------
    
    model = CLIPModel(
        position_list=position_list,
        im_weight_path=args.image_weight_path,
        image_encoder=args.image_encoder,
        num_classes=train_set.num_classes
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    loss_fn = CLIPLoss().to(args.device)
    
    min_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for image, meg, subject_idxs, y in tqdm(train_loader, desc="Train"):
            image, meg, subject_idxs, y = image.to(args.device), meg.to(args.device), subject_idxs.to(args.device), y.to(args.device)
            encoded_image, encoded_meg = model(image, meg, subject_idxs)
            
            loss = loss_fn(encoded_image, encoded_meg, y)
            y_pred = model.final_layer(encoded_meg)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for image, meg, subject_idxs, y in tqdm(val_loader, desc="Validation"):
            image, meg, subject_idxs, y = image.to(args.device), meg.to(args.device), subject_idxs.to(args.device), y.to(args.device)
            with torch.no_grad():
                encoded_image, encoded_meg = model(image, meg, subject_idxs)
                y_pred = model.final_layer(encoded_meg)
                acc = accuracy(y_pred, y)
                val_acc.append(acc.item())
            
            val_loss.append(loss_fn(encoded_image, encoded_meg, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_loss) < min_val_loss:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            min_val_loss = np.mean(val_loss)


if __name__ == "__main__":
    run()