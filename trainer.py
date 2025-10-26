from models.gtransformer import GTransformer, MaskedTransformer
from models.warmup import CosineWarmupScheduler
from utils.local import to_global

import torch
import warnings

from torch.amp import autocast
from lightning import LightningModule
from torch.nn.functional import normalize
try:
    from torch_scatter import scatter_mean, scatter_sum
except:
    warnings.warn('torch_scatter is not installed')


class ARGSModel(GTransformer):
    def __init__(self, label_smooth=0.1, pos_weight=1.0, scatter_bce=False, scatter_mse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
    
    def training_step(self, batch_p_gs, batch_idx) -> torch.Tensor:
        (
            prev_gs, prev_gs_split, next_gs, condition, 
            cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, 
            batch_p_gs, batch_n_gs, batch_size
        ) = batch_p_gs

        with autocast('cuda'):
            split, dense = self.forward(
                prev_gs, prev_gs[..., :3], condition, 
                cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, 
                None, None, 
                prev_gs_split
            )

            split_label = prev_gs_split[..., None].float().clip(min=self.hparams.label_smooth, max=1.0-self.hparams.label_smooth).cuda()
            pos_weight  = torch.tensor([self.hparams.pos_weight], device=split.device)
            reduction = 'none' if self.hparams.scatter_mse else 'mean'

            loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                split, split_label, 
                reduction=reduction, 
                pos_weight=pos_weight
            )

            loss_mse_xyz = torch.norm(dense[...,   :3]-next_gs[...,   :3], dim=1) # [T, 2, 14] -> [T, 2]
            loss_mse_opa = torch.nn.functional.l1_loss(torch.sigmoid(dense[...,  3:4]), next_gs[...,  3:4], reduction=reduction)
            loss_mse_rgb = torch.nn.functional.l1_loss(dense[...,  4:7], next_gs[...,  4:7], reduction=reduction)
            loss_mse_sca = torch.nn.functional.l1_loss(torch.exp(dense[..., 7:10]), next_gs[..., 7:10], reduction=reduction)
            loss_mse_qut = torch.sum(normalize(dense[..., 10:], dim=-1) * next_gs[..., 10:], dim=-1)
            loss_mse_qut = 1 - (loss_mse_qut ** 2)

            if self.hparams.scatter_mse: # Batch -> Mean
                loss_bce = scatter_mean(loss_bce.T, batch_p_gs).mean()

                loss_mse_xyz = scatter_mean(loss_mse_xyz.T, batch_n_gs).mean()
                loss_mse_opa = scatter_mean(loss_mse_opa.permute(2, 1, 0), batch_n_gs).mean()
                loss_mse_rgb = scatter_mean(loss_mse_rgb.permute(2, 1, 0), batch_n_gs).mean()
                loss_mse_sca = scatter_mean(loss_mse_sca.permute(2, 1, 0), batch_n_gs).mean()
                loss_mse_qut = scatter_mean(loss_mse_qut.T, batch_n_gs).mean()
            else:
                loss_mse_xyz = loss_mse_xyz.mean()
                loss_mse_qut = loss_mse_qut.mean()

            # weight is compute by std
            loss_mse = 1.0*loss_mse_xyz + 0.2*loss_mse_opa + 1.0*loss_mse_rgb + 1.0*loss_mse_sca + 0.4*loss_mse_qut
            
            loss = 0.7*loss_bce+0.3*loss_mse

        self.log_dict(
            {
                "train/loss_bce": loss_bce,
                "train/loss_mse": loss_mse,
                "train/loss_mse_xyz": loss_mse_xyz,
                "train/loss_mse_opa": loss_mse_opa,
                "train/loss_mse_rgb": loss_mse_rgb,
                "train/loss_mse_sca": loss_mse_sca,
                "train/loss_mse_qut": loss_mse_qut,
                "train/loss": loss,
            }, batch_size=batch_size,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        now_gs, next_gs_split, new_gs, condition, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, batch, batch_new_gs, batch_size = batch

        with autocast('cuda'):
            split, dense = self.forward(
                now_gs, now_gs[..., :3], condition, 
                cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, 
                None, None, 
                next_gs_split
            )

        bincount = torch.diff(cu_seqlens_gs)

        acc = (split>0)==next_gs_split[..., None]
        acc = scatter_sum(acc.long().T, batch)
        acc = acc / bincount
        acc = acc.mean()

        loss_mse_xyz = torch.nn.functional.l1_loss(dense[..., :3], new_gs[..., :3])
        loss_mse_opa = torch.nn.functional.l1_loss(torch.sigmoid(dense[..., 3:4]), new_gs[..., 3:4])
        loss_mse_rgb = torch.nn.functional.l1_loss(dense[..., 4:7], new_gs[..., 4:7])
        loss_mse_sca = torch.nn.functional.l1_loss(torch.exp(dense[..., 7:10]), new_gs[..., 7:10])
        loss_mse_qut = torch.nn.functional.l1_loss(normalize(dense[..., 10:], dim=-1), new_gs[..., 10:])

        self.log_dict(
            {
                "val/acc": acc,
                "val/loss_mse_xyz": loss_mse_xyz,
                "val/loss_mse_opa": loss_mse_opa,
                "val/loss_mse_rgb": loss_mse_rgb,
                "val/loss_mse_sca": loss_mse_sca,
                "val/loss_mse_qut": loss_mse_qut,
            }, batch_size=batch_size, sync_dist=True,
        )

    @torch.no_grad()
    def predict_step(self, batch, batch_idx) -> None:
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {
                "params": self.proj_g.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.proj_f.parameters(),
                "lr": 0.0001,
            },
            # {
            #     "params": self.f_uncond.parameters(),
            #     "lr": 0.001,
            # },
            {
                "params": self.ln_f.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.blocks.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.split_head.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.dense_head.parameters(),
                "lr": 0.0001,
            }
        ])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineWarmupScheduler(
                    optimizer, 
                    self.trainer.max_epochs//10, 
                    self.trainer.max_epochs,
                ),
            },
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is None:
            warnings.warn("No lr_schedulers found")
        elif isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                self.log_scheduler_lr(lr_scheduler)
        else:
            self.log_scheduler_lr(lr_schedulers)
    
    def log_scheduler_lr(self, lr_scheduler):
        for i in range(len(lr_scheduler.get_last_lr())):
            self.log(f"lr/last_lr_{i}", lr_scheduler.get_last_lr()[i])


class MaskedGSModel(MaskedTransformer):
    def __init__(self, label_smooth=0.1, pos_weight=1.0, scatter_bce=False, scatter_mse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def training_step(self, batch_p_gs, batch_idx) -> torch.Tensor:
        (
            prev_gs, _, _, condition, 
            cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, 
            batch_p_gs, batch_n_gs, batch_size
        ) = batch_p_gs

        ratio = max(0.1, min(0.5, self.trainer.current_epoch / self.trainer.max_epochs))

        mask = torch.rand(*prev_gs.shape[:-1], device=prev_gs.device) < ratio
        gsgt = prev_gs[mask]

        with autocast('cuda'):
            split, dense = self.forward(
                prev_gs, prev_gs[..., :3], condition, 
                cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, 
                None, None, 
                mask
            )

            split_label = mask[..., None].float().clip(min=self.hparams.label_smooth, max=1.0-self.hparams.label_smooth).cuda()
            pos_weight  = torch.tensor([self.hparams.pos_weight], device=split.device)
            reduction = 'none' if self.hparams.scatter_mse else 'mean'

            loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                split, split_label,
                reduction=reduction,
                pos_weight=pos_weight
            )

            loss_mse_xyz = torch.norm(dense[...,   :3]-gsgt[...,   :3], dim=1) # [T, 2, 14] -> [T, 2]
            loss_mse_opa = torch.nn.functional.l1_loss(torch.sigmoid(dense[...,  3:4]), gsgt[...,  3:4], reduction=reduction)
            loss_mse_rgb = torch.nn.functional.l1_loss(dense[...,  4:7], gsgt[...,  4:7], reduction=reduction)
            loss_mse_sca = torch.nn.functional.l1_loss(torch.exp(dense[..., 7:10]), gsgt[..., 7:10], reduction=reduction)
            loss_mse_qut = torch.sum(normalize(dense[..., 10:], dim=-1) * gsgt[..., 10:], dim=-1)
            loss_mse_qut = 1 - (loss_mse_qut ** 2)

            if self.hparams.scatter_mse: # Batch -> Mean
                loss_bce = scatter_mean(loss_bce.T, batch_p_gs).mean()

                loss_mse_xyz = scatter_mean(loss_mse_xyz.T, batch_n_gs).mean()
                loss_mse_opa = scatter_mean(loss_mse_opa.permute(2, 1, 0), batch_n_gs).mean()
                loss_mse_rgb = scatter_mean(loss_mse_rgb.permute(2, 1, 0), batch_n_gs).mean()
                loss_mse_sca = scatter_mean(loss_mse_sca.permute(2, 1, 0), batch_n_gs).mean()
                loss_mse_qut = scatter_mean(loss_mse_qut.T, batch_n_gs).mean()
            else:
                loss_mse_xyz = loss_mse_xyz.mean()
                loss_mse_qut = loss_mse_qut.mean()

            # weight is compute by std
            loss_mse = 1.0*loss_mse_xyz + 0.2*loss_mse_opa + 1.0*loss_mse_rgb + 1.0*loss_mse_sca + 0.4*loss_mse_qut
            
            loss = 0.2*loss_bce+0.7*loss_mse

        self.log_dict(
            {
                "train/loss_bce": loss_bce,
                "train/loss_mse": loss_mse,
                "train/loss_mse_xyz": loss_mse_xyz,
                "train/loss_mse_opa": loss_mse_opa,
                "train/loss_mse_rgb": loss_mse_rgb,
                "train/loss_mse_sca": loss_mse_sca,
                "train/loss_mse_qut": loss_mse_qut,
                "train/loss": loss,
            }, batch_size=batch_size,
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        pass

    @torch.no_grad()
    def predict_step(self, batch, batch_idx) -> None:
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {
                "params": self.proj_g.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.proj_f.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.f_uncond.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.ln_f.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.blocks.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.split_head.parameters(),
                "lr": 0.0001,
            },
            {
                "params": self.dense_head.parameters(),
                "lr": 0.0001,
            },
        ])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineWarmupScheduler(
                    optimizer, 
                    self.trainer.max_epochs//10, 
                    self.trainer.max_epochs,
                ),
            },
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is None:
            warnings.warn("No lr_schedulers found")
        elif isinstance(lr_schedulers, list):
            for lr_scheduler in lr_schedulers:
                self.log_scheduler_lr(lr_scheduler)
        else:
            self.log_scheduler_lr(lr_schedulers)
    
    def log_scheduler_lr(self, lr_scheduler):
        for i in range(len(lr_scheduler.get_last_lr())):
            self.log(f"lr/last_lr_{i}", lr_scheduler.get_last_lr()[i])
