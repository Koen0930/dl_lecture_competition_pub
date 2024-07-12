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

from src.datasets import ThingsMEGDataset
from src.clip import CLIPModel
from src.utils import set_seed


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )
    position_list = torch.load("/root/data/position_list.pt").to(args.device)

    # ------------------
    #       Model
    # ------------------
    # model = BasicConvClassifier(
    #     test_set.num_classes, test_set.seq_len, test_set.num_channels
    # ).to(args.device)
    # model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    model = CLIPModel(
        position_list=position_list,
        im_weight_path=args.image_weight_path,
        image_encoder=args.image_encoder,
        meg_encoder=args.meg_encoder,
        num_classes=1854
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"): 
        encoded_meg = model.meg_encoder(X.to(args.device), subject_idxs.to(args.device))  
        y_pred = model.final_layer(encoded_meg).detach().cpu()
        preds.append(y_pred)
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()