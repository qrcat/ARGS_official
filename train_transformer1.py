from models.dataset import MergeGaussianDataset, MergeGaussianDataModule
from models.transformer import ARGSTransformer

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info
from tqdm import tqdm, trange
from math import ceil

import torch

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default="log_transformer")
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--f32precision', type=str, default='medium', choices=['high', 'medium'])
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    # dataset
    parser.add_argument('--dataset', type=str, default="/mnt/private_rqy/gs_data/merge")
    parser.add_argument('--max_seq', type=int, default=16384)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--crop_min', type=int, default=None)
    parser.add_argument('--crop_max', type=int, default=None)
    parser.add_argument('--train_split', type=float, default=0.99)
    parser.add_argument('--load_in_memory', action='store_true')
    # checkpoint
    parser.add_argument('--checkpoint', type=str, default=None)
    # eval
    parser.add_argument('--eval', action='store_true')

    # args = parser.parse_args(['--load_in_memory'])
    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.f32precision)

    def init_dataset():
        return MergeGaussianDataModule(
            args.dataset, 
            batch_size=args.batch_size,
            permute=True,
            max_seq=args.max_seq,
            train_split=args.train_split,
            num_workers=args.num_workers,
            load_in_memory=args.load_in_memory,
        )

    dataset = init_dataset()

    if not args.eval:
        vqvae = ARGSTransformer()
    
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=1,
            save_last='link',
            verbose=True,
        )
        
        print(f"use devices{args.devices}")
        trainer = L.Trainer(
            default_root_dir=args.log_dir,
            max_epochs=args.max_epochs, 
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches,
            # device
            devices=args.devices,
            strategy="deepspeed_stage_2", 
            precision='16-mixed',
        )
        trainer.fit(vqvae, datamodule=dataset, ckpt_path=args.checkpoint)
    else:
        vqvae = SVQVAE.load_from_checkpoint(args.checkpoint)
        vqvae.freeze()

        trainer = L.Trainer(devices=args.devices)
        trainer.predict(vqvae, datamodule=dataset)
