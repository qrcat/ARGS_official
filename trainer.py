from models.gtransformer import GTransformer
from models.warmup import CosineWarmupScheduler
from utils.local import to_global

import torch
import warnings
import open_clip # for embedding
from torch.amp import autocast
from lightning import LightningModule
from torch.nn.functional import normalize
from torch_scatter import scatter_mean, scatter_sum


class ARGSModel(GTransformer, LightningModule):
    def __init__(self, label_smooth=0.1, pos_weight=2.0, scatter_bce=True, scatter_mse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        # tokenizer = open_clip.get_tokenizer('ViT-B-32')
        # clip_model.eval()
        # for param in clip_model.parameters():
        #     param.requires_grad = False

        # self.clip_model = {
        #     "tokenizer": tokenizer,
        #     "clip_model": clip_model,
        # }
        
    # def on_fit_start(self):
    #     self.clip_model['clip_model'].to(self.device)

    @torch.no_grad()
    def embed(self, text: list):
        tokenizer, clip_model = self.clip_model['tokenizer'], self.clip_model['clip_model']
        text = tokenizer(text).to(self.device)

        cast_dtype = clip_model.transformer.get_cast_dtype()

        x = clip_model.token_embedding(text).to(cast_dtype)

        x = x + clip_model.positional_embedding.to(cast_dtype)
        x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
        x = clip_model.ln_final(x)

        mask = text == 0
        bincount = torch.sum(~mask, dim=1)

        embedd = x[~mask]
        
        cu_seqlens_kv = torch.cumsum(torch.tensor([0]+bincount.tolist()), dim=0).to(dtype=torch.int32)
        # breakpoint()
        return embedd, cu_seqlens_kv.to(self.device)

    
    def training_step(self, batch_p_gs, batch_idx) -> torch.Tensor:
        prev_gs, prev_gs_split, next_gs, condition, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, batch_p_gs, batch_n_gs, batch_size = batch_p_gs

        with autocast('cuda'):
            self.hparams
            split, dense = self.forward(prev_gs, prev_gs[..., :3], condition, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, None, prev_gs_split)

            split_label = prev_gs_split[..., None].float().clip(min=self.hparams.label_smooth, max=1.0-self.hparams.label_smooth).cuda()

            if self.hparams.scatter_bce:
                loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    split, split_label, 
                    reduction='none', pos_weight=torch.tensor([self.hparams.pos_weight], device=split.device)
                )
                loss_bce = scatter_mean(loss_bce.T, batch_p_gs).mean()
            else:
                loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    split, split_label, 
                    pos_weight=torch.tensor([self.hparams.pos_weight], device=split.device)
                )
            
            if self.hparams.scatter_mse:
                loss_mse_xyz = torch.norm(dense[...,   : 3]-next_gs[...,   : 3], dim=1) # [T, 2, 14] -> [T, 2]
                loss_mse_xyz = scatter_mean(loss_mse_xyz.T, batch_n_gs).mean()       # [2, B]

                loss_mse_opa = torch.nn.functional.l1_loss(dense[...,  3: 4], next_gs[...,  3: 4], reduction='none')
                loss_mse_opa = scatter_mean(loss_mse_opa.permute(2, 1, 0), batch_n_gs).mean()

                loss_mse_rgb = torch.nn.functional.l1_loss(dense[...,  4: 7], next_gs[...,  4: 7], reduction='none')
                loss_mse_rgb = scatter_mean(loss_mse_rgb.permute(2, 1, 0), batch_n_gs).mean()

                loss_mse_sca = torch.nn.functional.l1_loss(dense[...,  7:10], next_gs[...,  7:10], reduction='none')
                loss_mse_sca = scatter_mean(loss_mse_sca.permute(2, 1, 0), batch_n_gs).mean()

                loss_mse_qut = torch.sum(dense[..., 10:] * next_gs[..., 10:], dim=-1)
                loss_mse_qut = 1 - (loss_mse_qut ** 2)
                loss_mse_qut = scatter_mean(loss_mse_qut.T, batch_n_gs).mean()
            else:
                loss_mse_xyz = torch.norm(dense[...,   : 3]-next_gs[...,   : 3], dim=1).mean()
                loss_mse_opa = torch.nn.functional.l1_loss(dense[...,  3: 4], next_gs[...,  3: 4])
                loss_mse_rgb = torch.nn.functional.l1_loss(dense[...,  4: 7], next_gs[...,  4: 7])
                loss_mse_sca = torch.nn.functional.l1_loss(dense[...,  7:10], next_gs[...,  7:10])
                loss_mse_qut = torch.sum(normalize(dense[..., 10: ], dim=-1) * next_gs[..., 10:], dim=-1)
                loss_mse_qut = 1 - (loss_mse_qut ** 2)
                loss_mse_qut = loss_mse_qut.mean()
            
            # weight is compute by std
            loss_mse = 1.0*loss_mse_xyz + 1.0*loss_mse_opa + 1.0*loss_mse_rgb + 1.0*loss_mse_sca + 1.0*loss_mse_qut
            
            loss = 1.0*loss_bce+1.0*loss_mse

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
        now_gs, next_gs_split, new_gs, embedd, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, batch, batch_new_gs, batch_size = batch

        # embedd, cu_seqlens_kv = self.embed(text)

        with autocast('cuda'):
            split, dense = self.forward(now_gs, now_gs[..., :3], embedd, cu_seqlens_gs, cu_seqlens_kv, maxseq_gs, maxseq_kv, None, next_gs_split)

        bincount = torch.diff(cu_seqlens_gs)

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
