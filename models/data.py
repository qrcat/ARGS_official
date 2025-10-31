from utils.local import to_local
from utils.quantize import Quantize

import json
import torch
import pickle
import numpy as np
import warnings

from pathlib import Path
from tqdm import tqdm
from joblib import delayed, Parallel
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule
from lightning.pytorch.utilities import rank_zero_only

class SimpleData(Dataset):
    quantize = Quantize()

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
        except FileNotFoundError:
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
        # new_gs = to_local(now_gs[next_gs_split], new_gs_gt)
        new_gs = self.quantize(new_gs_gt)
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
        prev_gs       = torch.cat([data[0] for data in batch])
        prev_gs_split = torch.cat([data[1] for data in batch])
        next_gs       = torch.cat([data[2] for data in batch])
        condition     = torch.cat([data[3] for data in batch])

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

        return prev_gs, prev_gs_split, next_gs, condition, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, batch_gs, batch_new_gs, batch_size

    @staticmethod
    def collate_fn_BLC(batch):
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

        prev_gs_batch = torch.zeros(batch_size, maxseq_gs, 14).to(prev_gs.device)
        condition_batch = torch.zeros(batch_size, maxseq_kv, condition.shape[-1]).to(condition.device)
        prev_gs_mask  = torch.zeros(batch_size, maxseq_gs, maxseq_gs, device=prev_gs.device, dtype=torch.bool)
        condition_mask = torch.zeros(batch_size, maxseq_gs, maxseq_kv, device=condition.device, dtype=torch.bool)
        for i in range(batch_size):
            prev_gs_batch[i, :cu_seqlens_gs[i+1]-cu_seqlens_gs[i], :] = prev_gs[cu_seqlens_gs[i]:cu_seqlens_gs[i+1], :]
            condition_batch[i, :cu_seqlens_kv[i+1]-cu_seqlens_kv[i], :] = condition[cu_seqlens_kv[i]:cu_seqlens_kv[i+1], :]
            prev_gs_mask[i, :cu_seqlens_gs[i+1]-cu_seqlens_gs[i], :cu_seqlens_gs[i+1]-cu_seqlens_gs[i]] = True
            condition_mask[i, :cu_seqlens_gs[i+1]-cu_seqlens_gs[i], :cu_seqlens_kv[i+1]-cu_seqlens_kv[i]] = True

        mask = torch.rand(*prev_gs_batch.shape[:-1], device=prev_gs.device) < 0.1
        for i in range(batch_size): mask[i, :cu_seqlens_gs[i+1]-cu_seqlens_gs[i]] = False

        recon = self.forward(prev_gs_batch, prev_gs_batch[..., :3], condition_batch, None, None, None, None, prev_gs_mask, condition_mask, mask)


        return now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, batch_gs, batch_new_gs, batch_size

class BatchData(Dataset):
    def __init__(
            self, 
            dir='data/airplane_pkl', pattern="*/point_cloud.pkl", meta_file='metas.json',
            post_load=True,   # don't load data at init
            apply_noise=True, # add noise to data
            no_check_meta_len=False,
        ):
        self.dir = Path(dir)

        self.apply_noise = apply_noise

        assert self.dir.exists()
        
        meta = self.dir / meta_file
        if meta.exists():
            # read metas
            with meta.open('r') as f:
                self.meta = json.load(f)
        else:
            raise FileNotFoundError(f'{meta} not found')
        
        self.data = self.meta['data']
        self.lens = self.meta['lens']

        self.cum_lens = np.cumsum([0] + self.lens)
        self.idx2bidx = torch.arange(
            len(self.lens)
        ).repeat_interleave(torch.tensor(self.lens, dtype=torch.int32))
    
    @rank_zero_only
    @staticmethod
    def build(dir: str, pattern: str, meta: str = 'metas.json', post_load: bool = False, apply_noise: bool = False):
        dir = Path(dir)
        data = []
        lens = []

        assert dir.exists()
        
        data = list(dir.glob(pattern))

        file = dir / meta

        meta = {'data': [], 'lens': [], 'size_per_sample': []}

        def get(path):
            data = SimpleData(path, apply_noise=apply_noise)
            return path.absolute().as_posix(), len(data)
        
        results = Parallel(n_jobs=3)(delayed(get)(path) for path in tqdm(data))

        data = [x[0] for x in results]
        lens = [x[1] for x in results]

        meta['data'] = data
        meta['lens'] = lens

        with file.open('w') as f:
            json.dump(meta, f)

        return meta, data, lens
    
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
            pattern="*/point_cloud.pkl",
            meta_file='metas.json',
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
            dataset = SimpleData(
                self.hparams.dir, 
                apply_noise=self.hparams.add_noise_on_data
            )
        else:
            dataset = BatchData(
                self.hparams.dir,
                self.hparams.pattern,
                self.hparams.meta_file,
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
    
    
