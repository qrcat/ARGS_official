from trainer import ARGSModel
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
    model = ARGSModel(14, 192, 12, 8, 0.1)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default="log")
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--f32precision', type=str, default='medium', choices=['high', 'medium'])
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    # dataset
    parser.add_argument('--dataset', type=str, default="../airplane_pkl")
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
        from models.data import SimpleData
        from models.gtransformer import GTransformer
        from utils.io import activated_gs2gs, save_ply
        from utils.local import to_global
        import torch


        dataset = SimpleData()

        model = GTransformer(14, 192, 12, 8, 0.1)
        model.load_state_dict(torch.load('log/lightning_logs/version_3/checkpoints/epoch=26-step=7182.ckpt')['state_dict'])
        model.eval()

        before = 8
        upstep = 20

        now_gs, next_gs_split, new_gs, embedd = dataset[before]

        now_gs_save = activated_gs2gs(now_gs)
        save_ply(f'{before}.ply', now_gs_save[..., :3], now_gs_save[..., 3:4], now_gs_save[..., 4:7], now_gs_save[..., 7:10], now_gs_save[..., 10:14])

        cu_seqlens_kv = torch.tensor([0, embedd.shape[0]], dtype=torch.int32)
        for i in range(upstep):
            cu_seqlens_gs = torch.tensor([0, now_gs.shape[0]], dtype=torch.int32)

            now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv = now_gs.cuda(), next_gs_split.cuda(), new_gs.cuda(), embedd.cuda(), cu_seqlens_gs.cuda(), cu_seqlens_kv.cuda()

            with torch.no_grad():
                split, dense = model(now_gs, now_gs[..., :3], embedd, cu_seqlens_gs, cu_seqlens_kv)

            split_mask = torch.squeeze(split > 0, dim=1)
            now_gs_split = now_gs[split_mask]
            
            new_gs_pred = dense[split_mask].view(-1, 2, 14)
            new_gs_pred = to_global(now_gs_split, new_gs_pred)
            
            if split_mask.all():
                now_gs = new_gs_pred.view(-1, 14)
            elif split_mask.any():
                now_gs = now_gs[~split_mask]
                now_gs = torch.cat([now_gs, new_gs_pred.view(-1, 14)], dim=0)
            else:
                i -= 1
                break

        now_gs_save = activated_gs2gs(now_gs)
        save_ply(f'{before}to{before+i+1}.ply', now_gs_save[..., :3], now_gs_save[..., 3:4], now_gs_save[..., 4:7], now_gs_save[..., 7:10], now_gs_save[..., 10:14])