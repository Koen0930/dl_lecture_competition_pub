import torch
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import torchvision.transforms as transforms
from src.datasets import ThingsMEGDataset


class preprocess():
    def cropping(self, meg):
        self.time = torch.arange(meg.shape[-1]) * 1400 / meg.shape[-1]
        self.image_time = torch.where((self.time > 93) & (self.time < 1307))[0]
        return meg[:, :, self.image_time]
    
    def scaler(self, meg):
        scaled_meg = []
        for waves in tqdm(meg):
            transformer = RobustScaler().fit(waves.T)
            waves_scaled = transformer.transform(waves.T).T
            waves_scaled = torch.tensor(waves_scaled, dtype=torch.float32)
            scaled_meg.append(waves_scaled)
        return torch.stack(scaled_meg)
    
    def clipping(self, meg):
        return torch.clamp(meg, -25, 20)
    
    def preprocess(self, meg):
        print("* cropping")
        meg = self.cropping(meg)
        print("* scaling")
        meg = self.scaler(meg)
        print("* clipping")
        meg = self.clipping(meg)
        return meg


def main():
        
    train_set = ThingsMEGDataset("train", "data")
    val_set = ThingsMEGDataset("val", "data")
    test_set = ThingsMEGDataset("test", "data")

    Preprocess = preprocess()
    preprocessed_val_X = Preprocess.preprocess(val_set.X)
    torch.save(preprocessed_val_X, "data/preprocessed_val_X.pt")
    preprocessed_test_X = Preprocess.preprocess(test_set.X)
    torch.save(preprocessed_test_X, "data/preprocessed_test_X.pt")
    preprocessed_train_X = Preprocess.preprocess(train_set.X)
    torch.save(preprocessed_train_X, "data/preprocessed_train_X.pt")
    

if __name__ == "__main__":
    main()
    
    