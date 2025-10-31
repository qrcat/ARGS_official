from utils.local import to_local
from utils.quantize import Quantize

import json
import torch
import pickle
import numpy as np
import warnings

from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from joblib import delayed, Parallel
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from lightning.pytorch.utilities import rank_zero_only

class CEData(Dataset):
    quantize = Quantize()

    def __init__(
            self, 
            path='data_block.pkl',
            padding_value=256,
            local=False, 
            apply_noise=True,  
            apply_quantize=True,
            return_indices=True
        ):
        self.padding_value = padding_value
        self.local = local
        self.apply_noise = apply_noise
        self.apply_quantize = apply_quantize
        self.return_indices = return_indices
        # 读取这个文件
        self.path = Path(path)
        with self.path.open('rb') as f:
            output_back = pickle.load(f)
        
        self.__dict__.update(output_back)
        self.mask_value = self.cumsum.repeat(self.count)

    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, idx):        
        sequence = torch.from_numpy(self.data[self.sequence[:idx+1]]).float() # [N, 14]
        position = sequence[..., :3]
        split_gs = torch.from_numpy(self.data[self.split_gs[:idx+1]]).float() # [N, 2, 14]
        split_bool = torch.from_numpy(self.split_bl[:idx+1])[..., None] # [N, 1]
        mask_value = torch.from_numpy(self.mask_value[:idx+1])
        
        masking_item = torch.arange(idx+1)>=mask_value[-1]
        
        # for the item > mask_value, set split_bool to False
        split_bool[masking_item] = False

        if not self.return_indices:
            raise NotImplementedError
            
        if self.apply_noise:
            raise NotImplementedError
        
        if self.local:
            split_gs = to_local(sequence, split_gs)
        
        if self.return_indices:
            sequence = self.quantize.get_indices(sequence)

            sequence[..., 3:4] += 256 * 1
            sequence[..., 4:7] += 256 * 2
            sequence[..., 7:10] += 256 * 3
            sequence[..., 10:14] += 256 * 4

            split_gs = self.quantize.get_indices(split_gs)
            # for the unsplit token, set the target to padding_value
            split_gs[~split_bool[..., 0]] = self.padding_value

            if idx >= len(self):
                warnings.warn(f'idx {idx} out of range')
        
        if not self.return_indices and self.apply_quantize:
            sequence = self.quantize(sequence)
            split_gs = self.quantize(split_gs)

        return sequence, position, split_gs, split_bool, mask_value

    def __repr__(self) -> str:
        return f"CEData({self.path}, seqlen:{len(self.sequence)})"

    def collate_fn(self, batch):
        return batch

    @staticmethod
    def _apply_noise(gs_params):
        noise_mask = torch.rand_like(gs_params[..., 0]) < 0.3

        volume_min = gs_params[noise_mask, 7:10].min(dim=1)[0]

        # now_gs[noise_mask, :3] += torch.randn_like(now_gs[noise_mask, :3]) * 0.001
        # or
        gs_params[noise_mask, :3] += torch.randn_like(gs_params[noise_mask, :3]) * volume_min[..., None] * 0.3
        
        gs_params[noise_mask, 3:4] *= (0.95+0.1*torch.randn_like(gs_params[noise_mask, 3:4]))
        gs_params[noise_mask, 3:4].clip_(0.0, 1.0)
    
        gs_params[noise_mask, 4:7] += torch.randn_like(gs_params[noise_mask, 4:7]) * 0.05
    

        gs_params[..., 7:10] *= (0.9+0.2*torch.randn_like(gs_params[..., 7:10]))
        gs_params[..., 7:10].clip_(min=1e-6)
        
        gs_params[..., 10:] *= (0.9+0.2*torch.randn_like(gs_params[..., 10:]))
        
        return gs_params
    
class BatchCEData(Dataset):
    def __init__(
            self, 
            dir='data/airplane_pkl', pattern="*/*block.pkl", meta_file='metas.json',
            max_len=16385,
            pre_load=False,
            save_meta=False,
            **args
        ):
        self.dir = Path(dir)
        assert self.dir.exists()

        self.max_len = max_len
        self.pre_load = pre_load
        self.cedata_args = args
        
        meta = self.dir / meta_file
        if meta.exists():
            # read metas
            with meta.open('r') as f:
                try:
                    self.meta = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    raise Exception(f"meta file {meta} is not valid")
        else:
            self.meta = sorted(self.dir.glob(pattern))

            if save_meta:
                self.save_meta(meta)
        
        if self.pre_load:
            self.data = Parallel(n_jobs=-1)(delayed(CEData)(path, **args) for path in tqdm(self.meta))
        else:
            self.data = None
    
    @rank_zero_only
    def save_meta(self, file: Path):
        with file.open('w') as f:
            json.dump([x.as_posix() for x in self.meta], f)
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        if self.data is None:
            return CEData(self.meta[idx], **self.cedata_args)[self.max_len]
        else:
            return self.data[idx][self.max_len]

    def collate_fn(self, batch):    
        sequence   = pad_sequence([data[0] for data in batch], True, 256*5)
        position   = pad_sequence([data[1] for data in batch], True, 0.0)
        split_gs   = pad_sequence([data[2] for data in batch], True, 256)
        split_bool = pad_sequence([data[3] for data in batch], True, False)
        mask_value = pad_sequence([data[4] for data in batch], True, sequence.shape[1])

        return sequence, position, split_gs, split_bool, mask_value

class BatchCEDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size=8, 
            num_workers=8, 
            shuffle=True, 
            split=[0.99, 0.01],
            **args,
        ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage):
        assert hasattr(self.hparams, 'path') or hasattr(self.hparams, 'dir')

        ce_data_args = {
            "local": self.hparams.local,
            "apply_noise": self.hparams.apply_noise,
            "apply_quantize": self.hparams.apply_quantize,
            "return_indices": self.hparams.return_indices,
        }
        
        if hasattr(self.hparams, 'path'):
            if Path(self.hparams.path).is_file():
                self.dataset = CEData(path=self.hparams.path, **ce_data_args)
        elif hasattr(self.hparams, 'dir'):
            self.dataset = BatchCEData(
                # 
                dir=self.hparams.dir,
                pattern=self.hparams.pattern,
                meta_file=self.hparams.meta_file,
                # 
                max_len=self.hparams.max_len,
                pre_load=self.hparams.pre_load,
                save_meta=self.hparams.save_meta,
                **ce_data_args,
            )
        
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = random_split(self.dataset, self.hparams.split)
        elif stage == "test" or stage is None:
            self.test_dataset = self.dataset
        elif stage == "predict":
            self.predict_dataset = self.dataset

    def collate_fn(self, batch):
        now_gs = torch.cat([data[0] for data in batch])
        next_gs_split = torch.cat([data[1] for data in batch])
        new_gs = torch.cat([data[2] for data in batch])

        cu_seqlens_gs = torch.cumsum(
            torch.tensor([0]+[data[0].shape[0] for data in batch]), dim=0
        ).to(dtype=torch.int32)

        embedd, cu_seqlens_kv = self.encode_text([data[3] for data in batch])
        
        return now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
            shuffle=self.hparams.shuffle
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )
    
if __name__ == '__main__':
    CEData()
