import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from topk.svm import SmoothTopkSVM
import torch.nn as nn

from src.datasets import ThingsMEGDataset
from src.models import CLIPModel, ClassifierModel
from src.utils import set_seed
from src.conformer import Conformer
from src.utils import MEGEncoder

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # すべてのGPUを指定

@hydra.main(version_base=None, config_path="configs", config_name="clip_retrain_config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="CLIP-retrain")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )
    position_list = torch.load("/root/data/position_list.pt").to(args.device)
    # ------------------
    #       Model
    # ------------------
    
    model = CLIPModel(
        position_list=position_list,
        im_weight_path=args.image_weight_path,
        image_encoder=args.image_encoder,
        num_classes=train_set.num_classes
    ).to(args.device)
    
    model.load_state_dict(torch.load("/root/outputs/2024-07-09/19-34-54/model_best.pt"))
    
    classifier = ClassifierModel(model).to(args.device)
    for name, param in list(classifier.named_parameters())[:-2]:
        param.requires_grad = False
            
    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    
    if args.loss == "topk":
        loss_fn = SmoothTopkSVM(n_classes=train_set.num_classes, alpha=None,
                                tau=1, k=10).cuda(args.device)
    elif args.loss == "ce":
        loss_fn = F.cross_entropy
    else:
        raise ValueError(f"Loss {args.loss} not supported")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        classifier.train()
        for X, y, subject in tqdm(train_loader, desc="Train"):
            X, y, subject = X.to(args.device), y.to(args.device), subject.to(args.device)
            
            y_pred = classifier(X, subject)
            
            loss = loss_fn(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        classifier.eval()
        for X, y, subject in tqdm(val_loader, desc="Validation"):
            X, y, subject = X.to(args.device), y.to(args.device), subject.to(args.device)
            with torch.no_grad():
                y_pred = classifier(X, subject)
            
            val_loss.append(loss_fn(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(classifier.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(classifier.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    classifier.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    classifier.eval()
    for X, subject in tqdm(test_loader, desc="Validation"):        
        preds.append(classifier(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()