from trainer import ARGSTrainer
from models.data import BatchDataModule
from models.gtransformer import GTransformer

from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import lightning as L

import argparse


def init_dataset():
    return BatchDataModule(
        args.dataset, 
        post_load=args.load_disk,
        batch_size=args.batch_size,            
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        split=[args.train_split, 1-args.train_split],
    )

def init_model():
    args_model = GTransformer(14, 192, 12, 8, 0.1)
    model = ARGSTrainer(args_model)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--f32precision', type=str, default='medium', choices=['high', 'medium'])
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    # dataset
    parser.add_argument('--dataset', type=str, default="/cluster/personal/ARGS/data/airplane_pkl")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--train_split', type=float, default=0.99)
    parser.add_argument('--load_disk', action='store_true', help='load data from disk online')
    parser.add_argument('--shuffle', action='store_true')
    # checkpoint
    parser.add_argument('--checkpoint', type=str, default=None)
    # eval
    parser.add_argument('--eval', action='store_true')
    # logger
    parser.add_argument('--logger', choices=['none', 'wandb', 'tensorboard', 'wd', 'tb'], default='none')

    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.f32precision)

    dataset = init_dataset()

    if not args.eval:
        if args.logger == 'none':
            logger = None
        elif args.logger in ['wandb', 'wd']:
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(project="ARGS")
        elif args.logger in ['tensorboard', 'tb']:
            from lightning.pytorch.loggers import TensorBoardLogger

            logger = TensorBoardLogger(save_dir=args.log_dir, name="args")

        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=1,
            save_last='link',
            verbose=True,
        )
        
        print(f"use devices{args.devices}")
        
        model = init_model()
        trainer = L.Trainer(
            default_root_dir=args.log_dir,
            max_epochs=args.max_epochs, 
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches,
            # device
            devices=args.devices,
            logger=logger,
            # strategy="deepspeed_stage_2",
        )
        trainer.fit(model, datamodule=dataset, ckpt_path=args.checkpoint)
    else:
        args_model = ARGSTransformer.load_from_checkpoint(args.checkpoint)
        args_model.freeze()

        trainer = L.Trainer(devices=args.devices)
        trainer.predict(args_model, datamodule=dataset)
