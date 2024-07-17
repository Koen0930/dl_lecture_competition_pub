import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torchmetrics import Accuracy
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import optuna
from topk.svm import SmoothTopkSVM

from src.datasets import ImageMEGDataset, BalancedClassBatchSampler
from src.clip import CLIPModel, CLIPLoss
from src.utils import set_seed

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

position_list = torch.load("data/position_list.pt").to("cuda:0")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = ImageMEGDataset("train", "data", transform=transform)
val_set = ImageMEGDataset("val", "data", transform=transform)

def train_and_evaluate(args, trial):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    loss_weight = trial.suggest_float('loss_weight', 0.5, 1.0)
    gat_dropout_rate = trial.suggest_float('gat_dropout_rate', 0.0, 1.0)
    pa_dropout_rate = trial.suggest_float('pa_dropout_rate', 0.0, 1.0)
    graph_k = trial.suggest_int('graph_k', 2, 271)
    
    # Transform
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # val_transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # Dataloader
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": True}
    
    print("* data loading")
    
    if args.balanced_sampling:
        train_sampler = BalancedClassBatchSampler(train_set, args.batch_size)
        val_sampler = BalancedClassBatchSampler(val_set, args.batch_size)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    
    
    print("* data loading done")

    # Model
    
    model = CLIPModel(
        position_list=position_list,
        im_weight_path=args.image_weight_path,
        image_encoder=args.image_encoder,
        meg_encoder=args.meg_encoder,
        num_classes=train_set.num_classes,
        gat_dropout=gat_dropout_rate,
        graph_k=graph_k,
        pa_dropout_rate=pa_dropout_rate,
    ).to(args.device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Start training
    accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(args.device)
    loss_fn = CLIPLoss().to(args.device)
    loss_fn2 = F.cross_entropy if args.loss == "ce" else SmoothTopkSVM(n_classes=train_set.num_classes, alpha=None, tau=1.0, k=10).cuda(args.device)
    
    max_val_acc = -float("inf")
    early_stopping_count = 0
    
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

        epoch_val_acc = np.mean(val_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {epoch_val_acc:.3f}")

        if epoch_val_acc > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = epoch_val_acc
            early_stopping_count = 0
        else:
            early_stopping_count += 1
 
        if early_stopping_count > 2:
            print(f"EPOCH{epoch}: early stopping")
            break
        if np.isnan(np.mean(val_loss)):
            print(f"NaN detected")
            break
        
    if args.use_wandb:
        wandb.log({"val_acc": max_val_acc})

    return max_val_acc

@hydra.main(version_base=None, config_path="configs", config_name="clip_config_params")
def main(args: DictConfig):
    if args.use_wandb:
        wandb.init(mode="online", project="CLIP_parmas")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: train_and_evaluate(args, trial), n_trials=200)

    print("Best hyperparameters: ", study.best_params)
    print("Best accuracy: ", study.best_value)

if __name__ == "__main__":
    main()
