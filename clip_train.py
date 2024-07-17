import os, sys
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from torchmetrics import Accuracy
import torch.nn as nn
from torch.optim import lr_scheduler
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from topk.svm import SmoothTopkSVM

from src.datasets import ImageMEGDataset, BalancedClassBatchSampler
from src.clip import CLIPModel, CLIPLoss
from src.utils import set_seed


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # すべてのGPUを指定

@hydra.main(version_base=None, config_path="configs", config_name="clip_config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="CLIP")
    
    # ------------------
    #       Transform
    # ------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # ランダムリサイズクロップ
        transforms.RandomHorizontalFlip(),  # ランダム水平反転
        transforms.RandomRotation(10),  # ランダム回転
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # カラージッター
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": True}
    
    print("* data loading")
    train_set = ImageMEGDataset("train", args.data_dir, transform=transform)
    val_set = ImageMEGDataset("val", args.data_dir, transform=transform)
    
    if args.balanced_sampling:
        train_sampler = BalancedClassBatchSampler(train_set, args.batch_size)
        val_sampler = BalancedClassBatchSampler(val_set, args.batch_size)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    
    position_list = torch.load("data/position_list.pt").to(args.device)
    
    print("* data loading done")
    # ------------------
    #       Model
    # ------------------
    
    model = CLIPModel(
        position_list=position_list,
        im_weight_path=args.image_weight_path,
        image_encoder=args.image_encoder,
        meg_encoder=args.meg_encoder,
        num_classes=train_set.num_classes
    ).to(args.device)

    
    # model.load_state_dict(torch.load("outputs/2024-07-17/07-14-39/model_best.pt"))

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 3. MultiStepLR
    # multistep_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 16, 17], gamma=0.1)

    # ------------------
    #   Start training
    # ------------------  
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    loss_fn = CLIPLoss().to(args.device)
    
    if args.loss == "topk":
        loss_fn2 = SmoothTopkSVM(n_classes=train_set.num_classes, alpha=None,
                                tau=1.0, k=10).cuda(args.device)
    elif args.loss == "ce":
        loss_fn2 = F.cross_entropy
    
    loss_weight = args.loss_weight
    min_val_acc = -float("inf")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for image, meg, subject_idxs, y in tqdm(train_loader, desc="Train"):
            image, meg, subject_idxs, y = image.to(args.device), meg.to(args.device), subject_idxs.to(args.device), y.to(args.device)
            encoded_image, encoded_meg = model(image, meg, subject_idxs)
            
            y_pred = model.final_layer(encoded_meg)
            
            # lossの計算
            clip_loss = loss_fn(encoded_image, encoded_meg, y)
            pred_loss = loss_fn2(y_pred, y)
            loss = loss_weight * clip_loss + (1 - loss_weight) * pred_loss
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
                # lossの計算
                clip_loss = loss_fn(encoded_image, encoded_meg, y)
                pred_loss = loss_fn2(y_pred, y)
                loss = loss_weight * clip_loss + (1 - loss_weight) * pred_loss
                
                acc = accuracy(y_pred, y)
                val_acc.append(acc.item())
            
            val_loss.append(loss.item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > min_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            min_val_acc = np.mean(val_acc)
        # multistep_scheduler.step()


if __name__ == "__main__":
    run()