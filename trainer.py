from models.gpt import GPT
from models.warmup import CosineWarmupScheduler
from utils.local import to_global
from utils.quaternion import normalize_quaternions

import torch
import torch.nn as nn
import warnings

from torch.amp import autocast
from lightning import LightningModule
from torch.nn.functional import normalize
try:
    from torch_scatter import scatter_mean, scatter_sum
except:
    warnings.warn('torch_scatter is not installed')



class ARGSModel(GPT):
    def __init__(self, warmup_rate=0.1, label_smooth=0.1, pos_weight=1.0, scatter_bce=False, scatter_mse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        sequence, position, split_gs, split_bool, mask_value = batch
        
        with autocast('cuda'):
            B, S, _, _ = split_gs.shape

            rets = self.forward(sequence, position, mask_value)
            split, dense = rets[..., :1], rets[..., 1:]
            
            dense = dense.view(B, S, 256, 2, 14).permute(0, 2, 1, 3, 4)

            split_label = split_bool.float().clip(min=self.hparams.label_smooth, max=1.0-self.hparams.label_smooth).cuda()
            pos_weight  = torch.tensor([self.hparams.pos_weight], device=split.device)

            loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                split, split_label,
                pos_weight=pos_weight
            )

            loss_ce = torch.nn.functional.cross_entropy(dense, split_gs, ignore_index=256)

            loss = loss_bce + loss_ce
            acc_split = ((split>0)==split_bool).float().mean()
            acc_dense = (dense.argmax(dim=1)==split_gs).float().mean()

        self.log_dict(
            {
                "train/loss_bce": loss_bce,
                "train/loss_ce": loss_ce,
                "train/acc_split": acc_split,
                "train/acc_dense": acc_dense,
            }
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
                "params": self.parameters(),
                "lr": 0.0001,
            },
        ])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineWarmupScheduler(
                    optimizer, 
                    int(self.trainer.max_epochs * self.hparams.warmup_rate), 
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
