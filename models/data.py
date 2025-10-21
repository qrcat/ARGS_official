from utils.local import to_local

import json
import torch
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule


class SimpleData(Dataset):
    def __init__(self, path='data.pkl', thres=0.05):
        # 读取这个文件
        with open(path, 'rb') as f:
            output_back = pickle.load(f)
        self.meta = output_back
        self.data = output_back['data']
        for l in range(self.meta['level']-1, -1, -1):
            if np.sum(self.meta[l][1])/len(self.meta[l][1]) > thres:
                self.level = l+1
                break

    def __len__(self):
        return self.level
    
    def __getitem__(self, idx):
        now_gs_index, next_gs_split, next_gs_index = self.meta[idx]

        now_gs = torch.from_numpy(self.data[now_gs_index]).float() # [N, 14]
        # add noise
        noise_mask = torch.rand_like(now_gs[..., 0]) < 0.3
        # breakpoint()

        volume_min = now_gs[noise_mask, 7:10].min(dim=1)[0]

        # now_gs[noise_mask, :3] += torch.randn_like(now_gs[noise_mask, :3]) * 0.001
        # or
        now_gs[noise_mask, :3] += torch.randn_like(now_gs[noise_mask, :3]) * volume_min[..., None] * 0.3
        
        now_gs[noise_mask, 3:4] *= (0.95+0.1*torch.randn_like(now_gs[noise_mask, 3:4]))
        now_gs[noise_mask, 3:4].clip_(0.0, 1.0)
    
        now_gs[noise_mask, 4:7] += torch.randn_like(now_gs[noise_mask, 4:7]) * 0.05
    

        now_gs[..., 7:10] *= (0.9+0.2*torch.randn_like(now_gs[..., 7:10]))
        now_gs[..., 7:10].clip_(min=1e-6)
        
        now_gs[..., 10:] *= (0.9+0.2*torch.randn_like(now_gs[..., 10:]))

        next_gs_split = torch.tensor(next_gs_split)
        
        new_gs_gt = torch.from_numpy(self.data[next_gs_index]).float()
        new_gs = to_local(now_gs[next_gs_split], new_gs_gt)
        
        # condition
        L = np.random.randint(5, 10)
        cond_embed = torch.randn(L, 192)

        return now_gs, next_gs_split, new_gs, cond_embed
    
    @staticmethod
    def collate_fn(batch):
        now_gs = torch.cat([data[0] for data in batch])
        next_gs_split = torch.cat([data[1] for data in batch])
        new_gs = torch.cat([data[2] for data in batch])
        embedd = torch.cat([data[3] for data in batch])
        
        cu_seqlens_gs = torch.cumsum(
            torch.tensor([0]+[data[0].shape[0] for data in batch]), dim=0
        ).to(dtype=torch.int32)
        cu_seqlens_kv = torch.cumsum(
            torch.tensor([0]+[data[3].shape[0] for data in batch]), dim=0
        ).to(dtype=torch.int32)

        return now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv

class BatchData(Dataset):
    def __init__(self, dir='data/airplane_pkl', post_load=True):
        self.dir = Path(dir)
        self.data = []
        self.lens = []

        assert self.dir.exists()
        
        data = list(self.dir.glob('*.pkl'))

        meta = self.dir / 'metas.json'
        if meta.exists():
            # read metas
            with meta.open('r') as f:
                self.meta = json.load(f)

            if len(self.meta['data']) != len(data):
                self.meta = None
            else:
                self.data = self.meta['data']
                self.lens = self.meta['lens']
        else:
            self.meta = None
        
        if not self.meta:
            self.meta = {'data': [], 'lens': [], 'size_per_sample': []}
            
            for path in tqdm(data):
                data = SimpleData(path)
                # size_per_l = []
                # for l in range(len(data)):
                #     size_per_l.append(data[l][1].shape[0])
                self.meta['data'].append(path.absolute().as_posix())
                self.meta['lens'].append(len(data))

                if not post_load:
                    self.data.append(data)
                else:
                    self.data.append(path)
            
            with meta.open('w') as f:
                json.dump(self.meta, f)
        else:
            self.data = self.meta['data']
        
        self.lens = self.meta['lens']

        self.cum_lens = np.cumsum([0] + self.lens)
        self.idx2bidx = torch.arange(
            len(self.lens)
        ).repeat_interleave(torch.tensor(self.lens, dtype=torch.int32))

    def __len__(self):
        return self.cum_lens[-1]
    
    def __getitem__(self, idx):
        b = self.idx2bidx[idx]
        i = idx - self.cum_lens[b]
        
        data = self.data[b]

        if isinstance(data, SimpleData):
            return data[i]
        else:
            return SimpleData(data)[i]            


class BatchDataModule(LightningDataModule):
    def __init__(self, dir='data/airplane_pkl', post_load=True, batch_size=8, num_workers=8, shuffle=True, split=[0.99, 0.01]):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage):
        dataset = BatchData(self.hparams.dir, post_load=self.hparams.post_load)
        
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = random_split(dataset, self.hparams.split)
        elif stage == "test" or stage is None:
            self.test_dataset = dataset
        elif stage == "predict":
            self.predict_dataset = dataset
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=SimpleData.collate_fn,
            shuffle=self.hparams.shuffle
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=SimpleData.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=SimpleData.collate_fn,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=SimpleData.collate_fn,
        )
    
    
