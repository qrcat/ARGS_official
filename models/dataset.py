from models.svqvae import SVQVAE
from utils.gaussian import norm_quats
from utils.quantize import Quantize

from torch.utils.data import Dataset, DataLoader, random_split
from typing import Union
from pathlib import Path
from tqdm import tqdm
from lightning import LightningDataModule
from math import log2
import json
import torch
import pickle
import warnings
import numpy as np
import torch.nn.functional as F


# ======================================================
# = this method is used for independent vqvqe training =
# ======================================================
class VQVAEDataset(Dataset):
    def __init__(
        self, 
        root: Union[str, Path],
        small_batch=4096,
        shuffle=True,
        permute=False,
        drop_last=False,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.small_batch = small_batch
        self.drop_last = drop_last
        
        data_files = list(root.glob('*.npz'))
        
        self.source = []
        self.target = []
        self.indice = []
        for data_file in data_files:
            data = np.load(data_file)
            source = torch.from_numpy(data['source']).to(torch.float)
            target = torch.from_numpy(data['target']).to(torch.float)
            indice = torch.arange(len(source))
            self.source.append(source)
            self.target.append(target)
            self.indice.append(indice)

        self.source = torch.concat(self.source)
        self.target = torch.concat(self.target)
        self.indice = torch.concat(self.indice)

        if permute:
            self.source = self.source.permute(0, 2, 1)

        if shuffle:
            reindex = torch.randperm(len(self.source))
            self.source = self.source[reindex]
            self.target = self.target[reindex]
            self.indice = self.indice[reindex]

        self.len = len(self.source) // small_batch if drop_last else len(self.source) // small_batch + 1

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        index_s = index * self.small_batch
        index_e = (index + 1) * self.small_batch
        return self.source[index_s:index_e], self.target[index_s:index_e], self.indice[index_s:index_e]

# ======================================================
# = this method is used for independent vqvqe training =
# ======================================================
class VQVAEDataModule(LightningDataModule):
    def __init__(
            self, 
            root, 
            max_seq=None, 
            batch_size=4, 
            num_workers=1, 
            permute=False, 
            shuffle=True, 
            split=(0.9, 0.1), 
            small_batch=4096, 
            drop_last=False
        ):
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage):
        if stage == "fit" or stage is None:
            dataset = VQVAEDataset(
                self.hparams.root,
                self.hparams.small_batch,
                self.hparams.shuffle,
                self.hparams.permute,
                self.hparams.drop_last
            )
            self.train_dataset, self.val_dataset = random_split(dataset, self.hparams.split)
        elif stage == "test" or stage is None:
            self.test_dataset = MergeGaussianDataset(
                self.hparams.root,
                self.hparams.permute,
                self.hparams.max_seq
            )
        elif stage == "predict":
            self.predict_dataset = MergeGaussianDataset(
                self.hparams.root,
                self.hparams.permute,
                self.hparams.max_seq
            )

    def train_collate_fn(self, batch):
        return torch.concat([d[0] for d in batch]), torch.concat([d[1] for d in batch]), torch.concat([d[2] for d in batch])
    
    def collate_fn(self, batch):
        padded_source = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch], batch_first=True)
        padded_target = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch], batch_first=True)
        padded_indice = torch.nn.utils.rnn.pad_sequence([data[2] for data in batch], batch_first=True)
        padded_mask = torch.nn.utils.rnn.pad_sequence([torch.ones(data[0].shape[0]) for data in batch], batch_first=True)

        return padded_source, padded_target, padded_indice, padded_mask.type(torch.bool)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_collate_fn,
            shuffle=self.hparams.shuffle
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_collate_fn,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )

# ===================================================
# = this method is used for sequence vqvqe training =
# ===================================================
class MergeGaussianDataset(Dataset):
    def __init__(
        self, 
        root: Union[str, Path],
        permute=False,
        max_seq=None,
        load_in_memory=True,
    ):
        if isinstance(root, str):
            root = Path(root)

        self.root = root
        self.data_files = sorted(root.glob('*.npz'))
        print(f"fetch {len(self.data_files)} files")
        
        self.permute = permute
        self.max_seq = max_seq

        self.quantize = Quantize()

        if load_in_memory:
            caches = sorted(self.root.glob('cache_*.pkl'))
            if caches:
                self.source = []
                self.target = []
                self.indice = []

                for cache in tqdm(caches, desc="fetch cache"):
                    with open(cache, "rb") as f:
                        source, target, indice = pickle.load(f)
                    
                    self.source += source
                    self.target += target
                    self.indice += indice
                    print(f"load cache from {cache}")
            else:
                try:
                    from joblib import Parallel, delayed

                    batch = 5000
                    epoch = np.ceil(len(self.data_files) / batch)
                    # only save by main process
                    for i in range(int(epoch)):
                        cache = self.root / f"cache_{i:04d}.pkl"

                        output = Parallel(n_jobs=32)(delayed(self.fetch_data)(i) for i in tqdm(self.data_files[i*batch:(i+1)*batch], desc='loading data'))

                        source, target, indice = zip(*output)
                        
                        with open(cache, "wb") as f: pickle.dump((source, target, indice), f)
                        print(f"save cache to {cache}")

                        del output, source, target, indice

                    exit(0)
                except:
                    warnings.warn("joblib is not installed, will not use parallel")
                    self.source = []
                    self.target = []
                    self.indice = []
                    for data_file in tqdm(self.data_files, desc='loading data'):
                        try:
                            source, target, indice = self.fetch_data(data_file)
                        
                            self.source.append(source)
                            self.target.append(target)
                            self.indice.append(indice)
                        except:
                            warnings.warn(f"error loading {data_file}")
                            continue
        else:
            self.source = None
            self.target = None
            self.indice = None

    def fetch_data(self, data_file):
        data = np.load(data_file)
        source = torch.from_numpy(data['source']).to(torch.float)
        target = torch.from_numpy(data['target']).to(torch.float)
        
        if self.permute:
            source = source.permute(0, 2, 1)
        
        if self.max_seq is not None:
            source = source[:self.max_seq]
            target = target[:self.max_seq]

        source = self.quantize(source)
        target = self.quantize(target)

        indice = torch.arange(len(source))

        return source, target, indice
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):
        if self.source is None or self.target is None or self.indice is None:
            data_file = self.data_files[index]
            source, target, indice = self.fetch_data(data_file)
        else:
            source, target, indice = self.source[index], self.target[index], self.indice[index]

        # don't do data enhance here, because it may change the merge list    
        return source, target, indice

# ===================================================
# = this method is used for sequence vqvqe training =
# ===================================================
class MergeGaussianDataModule(LightningDataModule):
    def __init__(
            self, 
            root, 
            max_seq=None,
            batch_size=None,
            num_workers=16,
            permute=False,
            shuffle=True,
            train_split=0.99,
            load_in_memory=True,
        ):
        super().__init__()

        self.save_hyperparameters()

        self.dataset = MergeGaussianDataset(
            self.hparams.root,
            self.hparams.permute,
            self.hparams.max_seq,
            self.hparams.load_in_memory,
        )

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = random_split(self.dataset, (self.hparams.train_split, 1-self.hparams.train_split))
        elif stage == "test" or stage is None:
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def collate_fn(self, batch):
        padded_source = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch], batch_first=True)
        padded_target = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch], batch_first=True)
        padded_indice = torch.nn.utils.rnn.pad_sequence([data[2] for data in batch], batch_first=True)
        padded_mask = torch.nn.utils.rnn.pad_sequence([torch.ones(data[0].shape[0]) for data in batch], batch_first=True)

        return padded_source, padded_target, padded_indice, padded_mask.type(torch.bool)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            shuffle=self.hparams.shuffle
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            # num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )

# ============================================================
# = this method is used for AR sequence Transformer training =
# ============================================================
class ARGaussianDataModule(LightningDataModule):
    def __init__(
            self, 
            root, 
            vqvae: str,
            max_seq=None,
            permute=False,

            shuffle=True,
            batch_size=2,
            num_workers=1,
            
        ):
        super().__init__()

        self.save_hyperparameters()

        self.vqvae = SVQVAE.load_from_checkpoint(vqvae)

    @torch.no_grad()
    def prepare_data(self):        
        
        vqvae.freeze()

        self.data = []

        dataset = MergeGaussianDataset(self.hparams.root, self.hparams.permute, self.hparams.max_seq)
        dataset = DataLoader(dataset, batch_size=2, collate_fn=MergeGaussianDataModule.collate_fn)
        for source, target, indices, masks in tqdm(dataset):
            source, target, indices = source.to(vqvae.device), target.to(vqvae.device), indices.to(vqvae.device)
            
            f_hat, label = vqvae.gaussian2idx(source, indices)
            
            for fhat, idx, cls, mask in zip(f_hat, indices, label, masks):
                fhat = fhat[mask]
                idx = idx[mask]
                cls = cls[mask]
                # add empty token in the input features
                fhat = torch.cat([torch.zeros_like(fhat[:1]), fhat], dim=0)
                # add sos token in the input sequence
                idx = torch.cat([torch.zeros_like(idx[:1]), idx+3], dim=0)
                # add eos token in the output sequence
                cls = torch.cat([cls+3, torch.ones_like(cls[:1])], dim=0)

                self.data.append((fhat.cpu(), idx.cpu(), cls.cpu()))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index] 

    def setup(self, stage):
        if stage == "fit" or stage is None:
            dataset = self
            self.train_dataset, self.val_dataset = random_split(dataset, (0.5, 0.5))
        elif stage == "test" or stage is None:
            self.test_dataset = self
        elif stage == "predict":
            self.predict_dataset = self
    
    def collate_fn(self, batch):
        padded_fhat = torch.nn.utils.rnn.pad_sequence([data[0] for data in batch], batch_first=True)
        # ============================================================================================================
        # add pad token
        padded_idx = torch.nn.utils.rnn.pad_sequence([data[1] for data in batch], batch_first=True, padding_value=2)
        padded_cls = torch.nn.utils.rnn.pad_sequence([data[2] for data in batch], batch_first=True, padding_value=2)
        # ============================================================================================================
        padded_mask = torch.nn.utils.rnn.pad_sequence([torch.ones(data[0].shape[0]) for data in batch], batch_first=True)

        return padded_fhat, padded_idx, padded_cls, padded_mask.type(torch.bool)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
            shuffle=self.hparams.shuffle
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
    dataset = ARGaussianDataModule("datasets")
    dataset.setup("fit")
    for data in dataset.train_dataloader():
        data
