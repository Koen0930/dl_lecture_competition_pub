import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from PIL import Image


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform = None) -> None:
        super().__init__()
        
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        # テキストファイルを1行ずつ読む
        with open(os.path.join(data_dir, f"{split}_image_paths.txt"), "r") as f:
            # 最後の改行を削除
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        for i, line in enumerate(lines):
            if "/" not in line:
                # lineの一番最後の_より前の文字を取得
                last_underscore_index = line.rfind('_')
                if last_underscore_index != -1:
                    line = f"{line[:last_underscore_index]}/{line}"
                lines[i] = line
        
        self.image_paths = lines
        self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
        assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i):
        if self.transform:
            self.X = Image.open(f"/root/data/Images/{self.image_paths[i]}")
            self.X = self.transform(self.X)
            return self.X, self.y[i]
        
    @property
    def height(self) -> int:
        # 画像の高さ
        return self.X.shape[2]
    
    @property
    def width(self) -> int:
        # 画像の幅
        return self.X.shape[3]


class ImageMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", transform = None) -> None:
        super().__init__()
        
        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.transform = transform
        
        # テキストファイルを1行ずつ読む
        with open(os.path.join(data_dir, f"{split}_image_paths.txt"), "r") as f:
            # 最後の改行を削除
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        for i, line in enumerate(lines):
            if "/" not in line:
                # lineの一番最後の_より前の文字を取得
                last_underscore_index = line.rfind('_')
                if last_underscore_index != -1:
                    line = f"{line[:last_underscore_index]}/{line}"
                lines[i] = line
        
        self.image_paths = lines
        
        self.meg = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
        assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i):
        if self.transform:
            self.image = Image.open(f"/root/data/Images/{self.image_paths[i]}")
            self.image = self.transform(self.image)
            return self.image, self.meg[i], self.subject_idxs[i], self.y[i]
        
    @property
    def height(self) -> int:
        # 画像の高さ
        return self.image.shape[2]
    
    @property
    def width(self) -> int:
        # 画像の幅
        return self.image.shape[3]
    
    @property
    def num_channels(self) -> int:
        return self.meg.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.meg.shape[2]