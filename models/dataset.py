from torch.utils.data import Dataset, DataLoader, random_split
from typing import Union
from pathlib import Path
from lightning import LightningDataModule
import json
import torch
import numpy as np


class ScaleGaussianDataset(Dataset):
    def __init__(self, root: Union[str, Path], split: str="train"):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        with (root / "data_list.json").open() as f:
            data_item = json.load(f)
        self.data_item = data_item

        # random_split(
        #     self.dataset, [0.8, 0.2]
        # )
        
        # normalize parameters
        self.normalized_params = {
            key: torch.from_numpy(value).to(torch.float)
            for key, value in np.load(root / "normalized_params.npz").items()
        }

    def __len__(self):
        return len(self.data_item)
    
    def __getitem__(self, index):
        path = self.root / self.data_item[index]
        data = np.load(path)
        data_dict = {}
        for key, value in data.items():
            value = torch.from_numpy(value).to(torch.float)

            std = self.normalized_params[f"{key}_std"]
            mean = self.normalized_params[f"{key}_mean"]
            
            data_dict[key] = value
            data_dict[f"{key}_normalized"] = (value - mean)/std

        return data_dict

class ScaleGaussianDataModule(LightningDataModule):
    def __init__(self, root, batch_size=1, num_workers=1):
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage):
        self.dataset = ScaleGaussianDataset(self.hparams.root)

        if stage == "fit" or stage is None:
            self.train_dataset = ScaleGaussianDataset(self.hparams.root, 'train')
            self.val_dataset = ScaleGaussianDataset(self.hparams.root, 'val')

        elif stage == "test" or stage is None:
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset
    
    def collate_fn(self, batch):
        data_dict = {}

        for key in batch[0].keys():
            data_dict[key] = torch.stack([b[key] for b in batch])
        
        for key in ['train_dataset', 'val_dataset', 'test_dataset', 'predict_dataset']:
            if hasattr(self, key): 
                data_dict.update(getattr(self, key).normalized_params)
            break

        return data_dict
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )

if __name__ == "__main__":
    dataset = ScaleGaussianDataModule("/mnt/e/code/ARGS/datasets")
    dataset.setup("fit")

    for data in dataset.train_dataloader():
        data
