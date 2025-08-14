from models.svqvae import SVQVAE
from models.dataset import VQVAEDataModule, MergeGaussianDataModule

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm, trange
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--gaussian_dim', type=int, default=14)
    parser.add_argument('--sin_dims', type=int, default=32)
    parser.add_argument('--z_channels', type=int, default=256)
    # vq config
    parser.add_argument('--n_embed', type=int, default=4096, help="number of vqvae embeddings")
    parser.add_argument('--embed_dim', type=int, default=192)
    # train
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default="log_svqvae")
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    # dataset
    parser.add_argument('--dataset', type=str, default="/mnt/private_rqy/gs_data/merge")
    parser.add_argument('--max_seq', type=int, default=16384)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--crop_min', type=int, default=0.01)
    parser.add_argument('--crop_max', type=int, default=1.00)
    parser.add_argument('--train_split', type=float, default=0.99)
    parser.add_argument('--load_in_memory', action='store_true')
    # checkpoint
    parser.add_argument('--checkpoint', type=str, default=None)
    # eval
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    dataset = MergeGaussianDataModule(
        args.dataset, 
        batch_size=args.batch_size,
        permute=True,
        max_seq=args.max_seq,
        train_split=args.train_split,
        num_workers=args.num_workers,
        load_in_memory=args.load_in_memory
    )

    if not args.eval:
        vqvae = SVQVAE(
            gaussian_dim=args.gaussian_dim,
            sin_dims=args.sin_dims,
            z_channels=args.z_channels,
            n_embed=args.n_embed,
            embed_dim=args.embed_dim,
            # data augmentation
            crop_min=args.crop_min,
            crop_max=args.crop_max,
        )

        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=1,
            save_last='link',
            verbose=True,
        )

        trainer = L.Trainer(
            default_root_dir=args.log_dir,
            max_epochs=args.max_epochs, 
            log_every_n_steps=1,
            devices=args.devices,
            val_check_interval=0.01,
            callbacks=[checkpoint_callback]

        )
        trainer.fit(vqvae, datamodule=dataset, ckpt_path=args.checkpoint)
    else:
        vqvae = SVQVAE.load_from_checkpoint(args.checkpoint)
        vqvae.freeze()

        trainer = L.Trainer(devices=args.devices)
        trainer.predict(vqvae, datamodule=dataset)
