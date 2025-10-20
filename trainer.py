from models.gtransformer import GTransformer
from models.warmup import CosineWarmupScheduler
from utils.local import to_global

import torch
import lightning as L
import warnings
from torch.amp import autocast
from torch_scatter import scatter_mean, scatter_sum


class ARGSTrainer(L.LightningModule):
    def __init__(self, model: GTransformer):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv = batch

        with autocast('cuda'):
            split, dense = self.model(now_gs, now_gs[..., :3], embedd, cu_seqlens_gs, cu_seqlens_kv, None, next_gs_split)

        bincount = torch.diff(cu_seqlens_gs)
        batch = torch.arange(
            len(bincount), device=bincount.device, dtype=torch.long
        ).repeat_interleave(bincount)

        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(split, next_gs_split[..., None].float().clip(min=0.1, max=1.0).cuda(), reduction='none')
        loss_bce = scatter_mean(loss_bce.T, batch).mean()

        acc = (split>0)==next_gs_split[..., None]
        acc = scatter_sum(acc.long().T, batch)
        acc = acc / bincount
        acc = acc.mean()

        loss_mse_xyz = torch.nn.functional.l1_loss(dense[..., :3], new_gs[..., :3])
        loss_mse_opa = torch.nn.functional.l1_loss(dense[..., 3:4], new_gs[..., 3:4])
        loss_mse_rgb = torch.nn.functional.l1_loss(dense[..., 4:7], new_gs[..., 4:7])
        loss_mse_sca = torch.nn.functional.l1_loss(dense[..., 7:10], new_gs[..., 7:10])
        loss_mse_qut = torch.nn.functional.l1_loss(dense[..., 10:], new_gs[..., 10:])
        # weight is compute by std
        loss_mse = 1.0*loss_mse_xyz + 0.02*loss_mse_opa + 0.02*loss_mse_rgb + 0.01*loss_mse_sca + 0.01*loss_mse_qut

        loss = 0.8*loss_bce+loss_mse

        self.log_dict(
            {
                "train/acc": acc,
                "train/loss_bce": loss_bce,
                "train/loss_mse": loss_mse,
                "train/loss_mse_xyz": loss_mse_xyz,
                "train/loss_mse_opa": loss_mse_opa,
                "train/loss_mse_rgb": loss_mse_rgb,
                "train/loss_mse_sca": loss_mse_sca,
                "train/loss_mse_qut": loss_mse_qut,
                "train/loss": loss,
            }, batch_size=bincount.shape[0],
        )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv = batch

        with autocast('cuda'):
            split, dense = self.model(now_gs, now_gs[..., :3], embedd, cu_seqlens_gs, cu_seqlens_kv, None, next_gs_split)

        bincount = torch.diff(cu_seqlens_gs)
        batch = torch.arange(
            len(bincount), device=bincount.device, dtype=torch.long
        ).repeat_interleave(bincount)

        acc = (split>0)==next_gs_split[..., None]
        acc = scatter_sum(acc.long().T, batch)
        acc = acc / bincount
        acc = acc.mean()

        loss_mse_xyz = torch.nn.functional.l1_loss(dense[..., :3], new_gs[..., :3])
        loss_mse_opa = torch.nn.functional.l1_loss(dense[..., 3:4], new_gs[..., 3:4])
        loss_mse_rgb = torch.nn.functional.l1_loss(dense[..., 4:7], new_gs[..., 4:7])
        loss_mse_sca = torch.nn.functional.l1_loss(dense[..., 7:10], new_gs[..., 7:10])
        loss_mse_qut = torch.nn.functional.l1_loss(dense[..., 10:], new_gs[..., 10:])

        self.log_dict(
            {
                "val/acc": acc,
                "val/loss_mse_xyz": loss_mse_xyz,
                "val/loss_mse_opa": loss_mse_opa,
                "val/loss_mse_rgb": loss_mse_rgb,
                "val/loss_mse_sca": loss_mse_sca,
                "val/loss_mse_qut": loss_mse_qut,
            }, batch_size=bincount.shape[0],
        )

    @torch.no_grad()
    def predict_step(self, batch, batch_idx) -> None:
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {
                "params": self.model.ffn.parameters(),
                "lr": 0.001,
            },
            {
                "params": self.model.blocks.parameters(),
                "lr": 0.0001,

            },
            {
                "params": self.model.uncond.parameters(),
                "lr": 0.00001,
            },
            {
                "params": self.model.split_head.parameters(),
                "lr": 0.001,
            },
            {
                "params": self.model.dense_head.parameters(),
                "lr": 0.001,
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

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        super().on_save_checkpoint(checkpoint)

        checkpoint['model_state_dict'] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        super().on_load_checkpoint(checkpoint)
    