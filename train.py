import configs

from trainer import ARGSModel
from models.data import BatchData, BatchDataModule
from models.gtransformer import GTransformer
from models.gpt.data import BatchCEDataModule

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only
import torch
import lightning as L
import datetime
import argparse

import os


def init_dataset(args):
    kw_args = dict(
        pattern=args.pattern, 
        meta_file=args.meta_file, 
        max_len=args.seqlen,
        pre_load=args.preload_in_memory, 
        save_meta=args.save_meta_in_disk, 
        padding_value=256,
        local=args.local_coords_data, 
        apply_noise=args.add_noise_on_data, 
        clip_outside=True,
        apply_quantize=False, 
        return_indices=True,
        # 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        split=[args.train_split, 1-args.train_split],
    )
    if os.path.isfile(args.dataset):
        kw_args['path'] = args.dataset
    else:
        kw_args['dir'] = args.dataset
    return BatchCEDataModule(
        **kw_args
    )

def init_model(args):
    config = getattr(configs, args.model)
    config['input_dim'] = 14
    config['output_dim'] = 2*14*256

    config['pos_weight'] = args.pos_weight
    config['warmup_rate'] = args.warmup_rate
    config['scatter_bce'] = args.scatter_bce
    config['scatter_mse'] = args.scatter_mse
    config['label_smooth'] = args.label_smooth

    model = ARGSModel(**config)

    return model


if __name__ == "__main__":
    # python train.py --dataset /mnt/private_rqy_tj/modelsplat_enhanced_block_pkl/ --pattern "airplane*/*block.pkl" --meta_file airplane.json --save_meta_in_disk --batch_size 4 --num_worke rs 64 --shuffle --logger wandb --model base_m_384 --devices 0 1 2 3 4 5 6 7 --accumulate_grad 2

    # NCCL_SOCKET_IFNAME=bond3 python train.py --dataset /mnt/private_rqy_tj/modelsplat_enhanced_block_pkl/ --pattern "*/*block.pkl" --preload_in_memory --save_meta_in_disk --batch_size 8 --num_workers 64 --shuffle --logger wandb --model base_m_768 --devices 0 1 2 3 4 5 6 7 --accumulate_grad 8 
    parser = argparse.ArgumentParser()

    parser_model = parser.add_argument_group('model')
    # model
    parser_model.add_argument('--model', type=str, default='base_s_192', choices=configs.__all__)
    parser_model.add_argument('--method', type=str, default='args', choices=['mask', 'args', 'mask2args', 'mask_ce', 'pca'])
    parser_model.add_argument('--pop_key', action='store_true')
    parser_model.add_argument('--warmup_rate', type=float, default=0.1)
    parser_model.add_argument('--label_smooth', type=float, default=0.0)
    parser_model.add_argument('--pos_weight', type=float, default=1.0, help='positive examples weight for bce loss.')
    parser_model.add_argument('--scatter_bce', action='store_true', help='scatter bce loss by batch.')
    parser_model.add_argument('--scatter_mse', action='store_true', help='scatter mse loss by batch.')
    # train
    parser_train = parser.add_argument_group('train')
    parser_train.add_argument('--epoch', type=int, default=100)
    parser_train.add_argument('--log_dir', type=str, default="log")
    parser_train.add_argument('--devices', type=int, nargs='+', default=[0])
    parser_train.add_argument('--precision', default='16-mixed')
    parser_train.add_argument('--strategy', default='auto', choices=['auto', 'ddp', 'ddp_find_unused_parameters_true', 'deepspeed_stage_1', 'deepspeed_stage_2'])
    parser_train.add_argument('--f32precision', type=str, default='medium', choices=['high', 'medium'])
    parser_train.add_argument('--gradient_clip_val', type=float)
    parser_train.add_argument('--accumulate_grad_batches', type=int, default=1)
    # dataset
    parser_datas = parser.add_argument_group('dataset')
    parser_datas.add_argument('--dataset', type=str, default=".")
    parser_datas.add_argument('--pattern', type=str, default="*block.pkl")
    parser_datas.add_argument('--meta_file', type=str, default="meta.json")
    parser_datas.add_argument('--seqlen', type=int, default=8192-1)
    parser_datas.add_argument('--train_split', type=float, default=1.0)
    parser_datas.add_argument('--local_coords_data', action='store_true')
    parser_datas.add_argument('--add_noise_on_data', action='store_true')
    parser_datas.add_argument('--preload_in_memory', action='store_true')
    parser_datas.add_argument('--save_meta_in_disk', action='store_true')

    parser_datas.add_argument('--batch_size', type=int, default=8)
    parser_datas.add_argument('--num_workers', type=int, default=32)
    parser_datas.add_argument('--shuffle', action='store_true')
    # checkpoint
    parser_check = parser.add_argument_group('checkpoint')
    parser_check.add_argument('--checkpoint', type=str, default=None)
    parser_check.add_argument('--resume_ckpt', type=str, default=None)
    parser_check.add_argument('--logger', choices=['none', 'wandb', 'tensorboard', 'wd', 'tb'], default='none')
    parser_check.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.f32precision)

    if not args.eval:
        dataset = init_dataset(args)

        if args.logger == 'none':
            logger = None
        elif args.logger in ['wandb', 'wd']:
            from lightning.pytorch.loggers import WandbLogger
            logger = WandbLogger(save_dir=args.log_dir, name=f"{datetime.datetime.now().strftime(r'%Y.%m.%d_%H:%M:%S')}", project=f'args_{args.method}')
        elif args.logger in ['tensorboard', 'tb']:
            from lightning.pytorch.loggers import TensorBoardLogger
            logger = TensorBoardLogger(save_dir=args.log_dir, name=f'args_{args.method}')

        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=1,
            # save_last='link',
            verbose=True,
            enable_version_counter=True,
        )
        
        print(f"use devices{args.devices}")
        
        model = init_model(args)
        trainer = L.Trainer(
            default_root_dir=args.log_dir,
            max_epochs=args.epoch,
            log_every_n_steps=1,
            val_check_interval=0.2,
            callbacks=[checkpoint_callback],
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.accumulate_grad_batches,
            # device
            devices=args.devices,
            logger=logger,
            precision='16-mixed',
            strategy=args.strategy,
        )

        if not args.checkpoint:
            model.apply(model.init_weights)
            print('model initialized')
        else:
            state_dict = torch.load(args.checkpoint, weights_only=False)['state_dict']
            rets = model.load_state_dict(state_dict, strict=False)
            print(rets)
            
        trainer.fit(model, datamodule=dataset, ckpt_path=args.resume_ckpt)
    else:
        from models.gpt.data import CEData
        from models.gtransformer import GTransformer
        from utils.quaternion import normalize_quaternions
        from utils.quantize import Quantize
        from utils.io import activated_gs2gs, save_ply
        from utils.local import to_global
        from utils.shs import RGB2SH


        import torch
        import configs

        args.batch_size = 1

        config = getattr(configs, args.model)

        model = init_model(args)
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        model.eval()
        model.cuda()

        if args.method == 'pca':
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler, MinMaxScaler

            scaler_standard = None
            pca = None
            scaler_minmax = None
        
            for level in range(len(dataset)-1, -1, -1):
                breakpoint()
                prev_gs, _, _, condition = dataset[level]
                
                logis = None
                # mask = torch.rand(*prev_gs.shape[:-1], device=prev_gs.device) < 0.25

                # now_gs_save = activated_gs2gs(prev_gs[mask])
                # save_ply(f'{level}-masked.ply', now_gs_save[..., :3], now_gs_save[..., 3:4], now_gs_save[..., 4:7], now_gs_save[..., 7:10], now_gs_save[..., 10:14])

                cu_seqlens_kv = torch.tensor([0, condition.shape[0]], dtype=torch.int32)
                cu_seqlens_gs = torch.tensor([0, prev_gs.shape[0]], dtype=torch.int32)

                prev_gs, condition, cu_seqlens_gs, cu_seqlens_kv = prev_gs.cuda(), condition.cuda(), cu_seqlens_gs.cuda(), cu_seqlens_kv.cuda()

                with torch.no_grad():
                    feat, logis, recon = model(
                        prev_gs, prev_gs[..., :3], condition, 
                        cu_seqlens_gs, cu_seqlens_kv, cu_seqlens_gs[-1], cu_seqlens_kv[-1], 
                        None, None, 
                        logis
                    )
                if scaler_standard is None:
                    scaler_standard = StandardScaler()
                    feat_scaled = scaler_standard.fit_transform(feat.cpu().numpy())
                else:
                    feat_scaled = scaler_standard.transform(feat.cpu().numpy())
                if pca is None:
                    pca = PCA(n_components=3)
                    feat_pca = pca.fit_transform(feat_scaled)
                else:
                    feat_pca = pca.transform(feat_scaled)
                if scaler_minmax is None:
                    scaler_minmax = MinMaxScaler()
                    feat_pca_scaled = scaler_minmax.fit_transform(feat_pca)
                else:
                    feat_pca_scaled = scaler_minmax.transform(feat_pca)

                # recon[..., 3:4] = torch.sigmoid(recon[...,  3:4])
                # recon[..., 4:7] = torch.from_numpy(RGB2SH(feat_pca_scaled))
                # recon[..., 7:10] = torch.exp(recon[..., 7:10])
                recon = prev_gs
                prev_gs[..., 4:7] = torch.from_numpy(RGB2SH(feat_pca_scaled))

                now_gs_save = activated_gs2gs(recon)
                save_ply(f'{level}-recon.ply', now_gs_save[..., :3], now_gs_save[..., 3:4], now_gs_save[..., 4:7], now_gs_save[..., 7:10], now_gs_save[..., 10:14])
                # breakpoint()
        else:
            quantize = Quantize()

            dataset = CEData(args.dataset)
            
            data = dataset[0]

            past_kvs = None
            
            inputs, positions = data[0][None], data[1][None]

            decode_sequence = torch.zeros(0, 14, device='cuda')

            for level in range(1, 10):
                with torch.no_grad():
                    B, S, _ = inputs.shape
                    inputs, positions = inputs.cuda(), positions.cuda()
                    rets, new_kvs = model(inputs.cuda(), positions.cuda(), past_kvs=past_kvs, use_cache=True)
                    logits, dense = rets[..., :1], rets[..., 1:]
                    
                    mask = logits > 0

                    decode_sequence = torch.cat([decode_sequence, inputs[~mask[..., 0]]], dim=0)

                    new_gs = dense.view(1, S, 256, 2, 14).permute(0, 2, 1, 3, 4).argmax(dim=1)
                    new_gs = new_gs[mask[..., 0]].view(-1, 14)

                    if not mask.any():
                        break
                    
                    # ===================================================================
                    # decode the sequence and output the ply file
                    # ===================================================================
                    now_gs = torch.cat([decode_sequence, new_gs], dim=0)
                    now_gs = quantize.dequantize(*now_gs.split([3, 1, 3, 3, 4], dim=-1))
                    now_gs = activated_gs2gs(*now_gs)
                    save_ply(f'decode_level_{level}.ply', *now_gs)
                    # ===================================================================

                    inputs = new_gs[None]
                    positions = quantize._dequantize_x(inputs[..., :3])

                    inputs[..., 3:4] += 256 * 1
                    inputs[..., 4:7] += 256 * 2
                    inputs[..., 7:10] += 256 * 3
                    inputs[..., 10:14] += 256 * 4
                    
                    past_kvs = new_kvs
            
