from utils.local import to_local

import json
import torch
import pickle
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule


class SimpleData(Dataset):
    def __init__(self, path='data.pkl', thres=0.05, apply_noise=True):
        self.apply_noise = apply_noise
        # 读取这个文件
        self.path = Path(path)
        with self.path.open('rb') as f:
            output_back = pickle.load(f)
        self.meta = output_back
        self.data = output_back['data']
        for l in range(self.meta['level']-1, -1, -1):
            if len(self.meta[l][1]) < 16384 and np.sum(self.meta[l][1])/len(self.meta[l][1]) > thres:
                self.level = l+1
                break
        try:
            self.cond = np.load(self.path.parent / 'feature.npy')
        except:
            warnings.warn(f'feature.npy not found in {self.path.parent}')
            self.cond = np.random.randn(8, 10, 768) # fake condition

    def __len__(self):
        return self.level
    
    def __getitem__(self, idx):
        now_gs_index, next_gs_split, next_gs_index = self.meta[idx]

        now_gs = torch.from_numpy(self.data[now_gs_index]).float() # [N, 14]
        
        if self.apply_noise:            
            # add noise
            now_gs = self._apply_noise(now_gs)

        next_gs_split = torch.tensor(next_gs_split)
            
        new_gs_gt = torch.from_numpy(self.data[next_gs_index]).float()
        new_gs = to_local(now_gs[next_gs_split], new_gs_gt)
            
        # condition
        cond = self.cond[np.random.randint(0, 8, 1)][0]
        cond = torch.from_numpy(cond).float()

        return now_gs, next_gs_split, new_gs, cond
    
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

        bincount_gs = torch.diff(cu_seqlens_gs)
        bincount_kv = torch.diff(cu_seqlens_kv)
        
        maxseq_gs = bincount_gs.max()
        maxseq_kv = bincount_kv.max()

        batch_gs = torch.arange(
            len(bincount_gs), device=bincount_gs.device, dtype=torch.long
        ).repeat_interleave(bincount_gs)

        bincount_new_gs = torch.tensor([data[2].shape[0] for data in batch])
        batch_new_gs = torch.arange(
            len(bincount_new_gs), device=bincount_new_gs.device, dtype=torch.long
        ).repeat_interleave(bincount_new_gs)

        batch_size = len(bincount_gs)

        return now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, batch_gs, batch_new_gs, batch_size

class BatchData(Dataset):
    def __init__(
            self, 
            dir='data/airplane_pkl', pattern="*/point_cloud.pkl", 
            post_load=True,   # don't load data at init
            apply_noise=True, # add noise to data
            no_check_meta_len=False,
        ):
        self.dir = Path(dir)
        self.data = []
        self.lens = []

        self.apply_noise = apply_noise

        assert self.dir.exists()
        
        data = list(self.dir.glob(pattern))

        meta = self.dir / 'metas.json'
        if meta.exists():
            # read metas
            with meta.open('r') as f:
                self.meta = json.load(f)

            if not no_check_meta_len and len(self.meta['data']) != len(data):
                self.meta = None
            else:
                self.data = self.meta['data']
                self.lens = self.meta['lens']
        else:
            self.meta = None
        
        if not self.meta or not post_load:
            self.meta = {'data': [], 'lens': [], 'size_per_sample': []}
            
            for path in tqdm(data):
                data = SimpleData(path, apply_noise=self.apply_noise)
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
            return SimpleData(data, apply_noise=self.apply_noise)[i]

class BatchDataModule(LightningDataModule):
    def __init__(
            self, 
            dir='data/airplane_pkl', 
            add_noise_on_data=True,
            no_check_meta_len=False,
            post_load=True, 
            batch_size=8, 
            num_workers=8, 
            shuffle=True, 
            split=[0.99, 0.01],
        ):
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage):
        if Path(self.hparams.dir).is_file():
            dataset = SimpleData(self.hparams.dir, apply_noise=self.hparams.add_noise_on_data)
        else:
            dataset = BatchData(
                self.hparams.dir, 
                post_load=self.hparams.post_load, 
                apply_noise=self.hparams.add_noise_on_data,
                no_check_meta_len=self.hparams.no_check_meta_len,
            )
        
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = random_split(dataset, self.hparams.split)
        elif stage == "test" or stage is None:
            self.test_dataset = dataset
        elif stage == "predict":
            self.predict_dataset = dataset
    
    @torch.no_grad()
    def encode_text(self, text):
        text = self.tokenizer(text).to(self.device)

        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = self.clip_model.ln_final(x)

        mask = text == 0
        bincount = torch.sum(~mask, dim=1)

        embedd = x[~mask]
        
        cu_seqlens_kv = torch.cumsum(torch.tensor([0]+bincount.tolist()), dim=0).to(dtype=torch.int32)
        
        return embedd, cu_seqlens_kv

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
    
    
