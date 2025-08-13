from models.svqvae import SVQVAE
from models.dataset import VQVAEDataModule, MergeGaussianDataModule

import lightning as L
from tqdm import tqdm, trange
import torch
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main():
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--log_dir', type=str, default="log_svqvae")
    # dataset
    parser.add_argument('--dataset', type=str, default="/data/workspace/ARGS/datasets")
    parser.add_argument('--batch_size', type=int, default=2)
    # checkpoint
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    dataset = MergeGaussianDataModule(
        args.dataset, 
        batch_size=args.batch_size,
        permute=True
    )

    if not args.eval:
        vqvae = SVQVAE()

        trainer = L.Trainer(
            default_root_dir=args.log_dir, 
            max_epochs=args.max_epochs, 
            log_every_n_steps=2
        )
        trainer.fit(vqvae, datamodule=dataset, ckpt_path=args.checkpoint)
        # trainer.fit(vqvae, datamodule=dataset, ckpt_path="log_svqvae/lightning_logs/version_4/checkpoints/epoch=898-step=899.ckpt")
    else:
        vqvae = SVQVAE.load_from_checkpoint(args.checkpoint)
        vqvae.freeze()

        trainer = L.Trainer()
        trainer.predict(vqvae, datamodule=dataset)
