from .basic import BasicVAE, ScaleGaussianSplat
from .decoder import resnet34_encoder, resnet34_decoder
from .sin_encoder import GaussianSinEncoder, SinPositionEncoding
from .warmup import CosineWarmupScheduler
from utils.args import ARDecoder

from vector_quantize_pytorch import ResidualVQ
from tqdm import tqdm

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from math import log2

import torch

import warnings


class SVQVAE(BasicVAE):
    def __init__(
        self,
        gaussian_dim=14, sin_dims=32, z_channels=512,
        dropout=0.0,
        path_nums=4,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        weight_x=10.0, 
        weight_o=2.0, 
        weight_f=1.0, 
        weight_s=10.0, 
        weight_q=1.0, 
        weight_vq=0.5,
        # rvq config
        embed_dim = 192,         # vq embed dim
        n_embed =  4096,         # vq num embeddings
        embed_levels = 2,        # rvq levels
        embed_loss_weight = 1.0,
        stochasticity = 0.1,
        embed_share = True,      # share embeddings across rvq levels
        code_decay = 0.99,       # code decay for vq
    ):
        super(SVQVAE, self).__init__()
        self.save_hyperparameters()

        self.scale_gs = ScaleGaussianSplat()
        self.sin_encoder = GaussianSinEncoder(sin_dims)
        
        # self.encoder = ResNet18Encoder(142, z_channels)
        # self.decoder = ResNet18Decoder(z_channels, gaussian_dim)
        input_dim = self.sin_encoder.out_dim*2
        output_dim = gaussian_dim*2
        self.encoder = resnet34_encoder(input_dim, embed_dim)
        self.decoder = resnet34_decoder(embed_dim, output_dim)
        
        self.sin_position_encoding = SinPositionEncoding(embed_dim, max_sequence_length=66536)

        self.pre_quant = torch.nn.Linear(embed_dim, embed_dim)
        self.post_quant = torch.nn.Linear(embed_dim, embed_dim)

        self.residual_vq = ResidualVQ(
            dim = embed_dim,
            codebook_size = n_embed,               # codebook size
            num_quantizers = embed_levels,
            commitment_weight = embed_loss_weight, # the weight on the commitment loss
            stochastic_sample_codes = True,
            sample_codebook_temp = stochasticity,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook = embed_share,         # whether to share the codebooks for all quantizers or not
            decay = code_decay,
        )

    def training_step(self, batch, batch_idx):
        source = batch[0]
        # target = batch[1] # unused
        indice = batch[2]
        bhmask = batch[3]
        
        pred, _, loss_vq = self.forward(source, indice)

        loss_rc, loss_disp = self.reconstruct_loss(pred, source, indice, bhmask, weights=[self.hparams.weight_x, self.hparams.weight_o, self.hparams.weight_f, self.hparams.weight_s, self.hparams.weight_q])

        vq_weight = self.hparams.weight_vq if self.current_epoch > 100 else 0
        loss = loss_rc + vq_weight * loss_vq.sum()

        self.log_dict({
            "loss/loss": loss,
            # reconstruct
            "recon/rc": loss_rc.item(),
            "recon/x": loss_disp[0],
            "recon/o": loss_disp[1],
            "recon/f": loss_disp[2],
            "recon/s": loss_disp[3],
            "recon/q": loss_disp[4],
        })
        # vq
        for si in range(len(loss_vq[0])): self.log(f"vq/loss_{si}", loss_vq[0, si])

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        source = batch[0]
        # target = batch[1] # unused
        indice = batch[2]
        bhmask = batch[3]
        
        pred, _, _ = self.forward(source, indice)

        metrics = self.eval_reconstruct(pred, source, indice, bhmask)

        self.log_dict({
            f"metrics/{key}": value
            for key, value in metrics.items()
        })

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        source, target, indices, batch = batch
        gaussianes, _, _ = self.forward(source, indices)
        gaussianes = self.to_activated(gaussianes, indices)
        for index, (gaussian, indice) in enumerate(zip(gaussianes, indices)):
            ardecoder = ARDecoder(gaussian.shape[0]*2, device=self.device)
            for i, gs in enumerate(tqdm(gaussian)):
                if i < 1024: 
                    top_k = i // 4 if i > 128 else 32
                    solve_by = 'scale'
                elif i < 4096: 
                    top_k = 128
                    solve_by = 'dist'
                else:
                    top_k = 1 
                    solve_by = 'none'
                ardecoder.update(gs, top_k=top_k, solve_by=solve_by)
                if len(ardecoder) in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
                    ardecoder.save_ply(f'output/decode-{index}-{len(ardecoder)}.ply')

    # ========================`forward` is only used for training========================
    def forward(self, gaussian, indice):   # -> rec_B3HW, idx_N, loss
        gaussian = self.scale_gs.forward(gaussian)    
        gaussian = self.sin_encoder(gaussian)
        gaussian = gaussian.flatten(-2, -1)

        embed = self.encoder(gaussian.permute(0, 2, 1))
        embed = self.pre_quant(embed.permute(0, 2, 1))

        if self.current_epoch <= 3:
            f_hat, indices, commit_loss = self.residual_vq(embed, freeze_codebook = True)
        else:
            f_hat, indices, commit_loss = self.residual_vq(
                embed, freeze_codebook = False if not self.training else True
            )

        pos_embed = self.sin_position_encoding.query(indice)

        output = self.decoder(self.post_quant(f_hat+pos_embed).permute(0, 2, 1))
        output = output.permute(0, 2, 1)

        B, S, C = output.shape
        output = output.reshape(B, S, 2, C//2)

        return output, indices, commit_loss
    # ===================================================================================

    def gaussian2idx(self, gaussian, indice):
        gaussian = self.scale_gs.forward(gaussian)    
        gaussian = self.sin_encoder(gaussian)
        gaussian = gaussian.flatten(-2, -1)

        embed = self.encoder(gaussian.permute(0, 2, 1))
        embed = self.pre_quant(embed.permute(0, 2, 1))

        f_hat, indices, _ = self.residual_vq(embed, freeze_codebook = False)
        return f_hat, indices
    
    def idx2gaussian(self, f_hat: torch.Tensor, indice: torch.Tensor):
        pos_embed = self.sin_position_encoding.query(indice)

        output = self.decoder(self.post_quant(f_hat+pos_embed).permute(0, 2, 1))
        output = output.permute(0, 2, 1)

        B, S, C = output.shape
        output = output.reshape(B, S, 2, C//2)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineWarmupScheduler(optimizer, 1000, 10000),
                # "monitor": "metric_to_track",
                # "frequency": "indicates how often the metric is updated",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
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
            self.log(f"lr/last_lr_{i}", lr_scheduler.get_last_lr()[i], sync_dist=True)
