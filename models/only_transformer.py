from .basic import BasicVAE, ScaleGaussianSplat
from .warmup import CosineWarmupScheduler
from .decoder import resnet34_encoder, resnet34_decoder
from .sin_encoder import GaussianSinEncoder, SinPositionEncoding
from utils.args import ARDecoder

from vector_quantize_pytorch import ResidualVQ
from tqdm import tqdm
import numpy as np
import torch

import warnings
import os

class OnlycTransformer(BasicVAE):
    def __init__(
        self,
        gaussian_dim=14, # input dims
        sin_dims=32,     # sin embedding dims
        z_channels=256,  # latent dims
        # ==================================================================
        # =                          loss weights                          =
        # ==================================================================
        weight_x=10.0, 
        weight_o=2.0, 
        weight_f=1.0, 
        weight_s=10.0, 
        weight_q=1.0,
        # ==================================================================
        # =                           rvq config                           =
        # ==================================================================
        embed_dim = 192,         # vq embed dim
        n_embed = 16384,         # vq num embeddings
        embed_levels = 2,        # rvq levels
        embed_loss_weight = 0.2, # embed loss weight
        stochasticity = 0.1,     # stochasticity
        embed_share = True,      # share embeddings across rvq levels
        code_decay = 0.9,        # code decay for vq
        kmeans_iters = 100,      # kmeans iterations
        # ==================================================================
        # =                   training data augmentation                   =
        # ==================================================================
        crop_min = None, # crop min length
        crop_max = None, # crop max length
    ) -> None:
        super(SVQVAE, self).__init__()
        self.save_hyperparameters()

        self.scale_gs = ScaleGaussianSplat()
        self.sin_encoder = GaussianSinEncoder(sin_dims)
        # combine two gaussian into one token
        input_dim = self.sin_encoder.out_dim*2
        output_dim = gaussian_dim*2
        # self.encoder = resnet34_encoder(input_dim, z_channels)
        self.decoder = resnet34_decoder(z_channels, output_dim)
        
        self.pre_quant = torch.nn.Linear(input_dim, embed_dim)
        self.post_quant = torch.nn.Linear(embed_dim, z_channels)

        self.sin_position_encoding = SinPositionEncoding(z_channels, max_sequence_length=66536)

        print("kmeans iters: ", kmeans_iters)
        self.residual_vq = ResidualVQ(
            # codebook config
            dim = embed_dim,                        # code dimension
            codebook_size = n_embed,                # codebook size
            num_quantizers = embed_levels,          # number of quantizers
            shared_codebook = embed_share,          # whether to share the codebooks for all quantizers or not
            # init the codebook
            kmeans_init = True,
            kmeans_iters = kmeans_iters,            # number of kmeans iterations to calculate the centroids for the codebook on init
            # commitment loss
            commitment_weight = embed_loss_weight,  # the weight on the commitment loss
            # sample and update
            stochastic_sample_codes = True,
            sample_codebook_temp = stochasticity,   # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            decay = code_decay,                     # the exponential moving average decay, lower means the dictionary will change faster
        )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        source = batch[0]
        # target = batch[1] # unused
        indice = batch[2]
        bhmask = batch[3]

        if self.hparams.crop_min is not None and self.hparams.crop_max is not None:
            crop_win = 0.5 * np.cos(self.global_step / 100 * np.pi) + 0.5
            crop_min = self.hparams.crop_max - (self.hparams.crop_max-self.hparams.crop_min) * crop_win
            crop_max = self.hparams.crop_max
            crop_len = torch.rand((1,)) * (crop_max-crop_min) + crop_min
            crop_len = source.size(1) * crop_len

            self.log_dict({
                "crop/win": crop_win, 
                "crop/min": crop_min, 
                "crop/max": crop_max, 
                "crop/len": crop_len
            })

            crop_len = crop_len.long()            
            source = source[:, :crop_len]
            # target = target[:, :crop_len]
            indice = indice[:, :crop_len]
            bhmask = bhmask[:, :crop_len]
        
        pred, _, loss_vq = self.forward(source, indice)

        loss_rc, loss_disp = self.reconstruct_loss(pred, source, indice, bhmask, weights=[self.hparams.weight_x, self.hparams.weight_o, self.hparams.weight_f, self.hparams.weight_s, self.hparams.weight_q])

        vq_weight = min(self.current_epoch/(self.trainer.max_epochs/10), 1.0)
        loss = loss_rc + vq_weight * loss_vq.mean()
        self.log("train/vq_weight", vq_weight)

        self.log_dict({
            "loss/loss": loss,
            # reconstruct
            "loss/reconstruct": loss_rc.item(),
            "loss/xyz": loss_disp[0],
            "loss/opacity": loss_disp[1],
            "loss/feature": loss_disp[2],
            "loss/scale": loss_disp[3],
            "loss/quaternion": loss_disp[4],
        })
        # vq
        for si in range(len(loss_vq[0])): self.log(f"loss/vq_{si}", loss_vq[0, si])

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx) -> None:
        source = batch[0]
        # target = batch[1] # unused
        indice = batch[2]
        bhmask = batch[3]
        
        pred, _, _ = self.forward(source, indice)

        metrics = self.eval_reconstruct(pred, source, indice, bhmask)

        self.log_dict({
            f"metrics/{key.replace('/', '_')}": value
            for key, value in metrics.items()
        }, sync_dist=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        # create output dir
        os.path.mkdir('output', exist_ok=True)
        # predict
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
                # save ply
                if len(ardecoder) in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
                    ardecoder.save_ply(f'output/decode-{index}-{len(ardecoder)}.ply')

    # ========================`forward` is only used for training========================
    def forward(self, gaussian, indice):   # -> rec_B3HW, idx_N, loss
        gaussian = self.scale_gs.forward(gaussian)
        gaussian = self.sin_encoder(gaussian)
        gaussian = gaussian.flatten(-2, -1)

        embed = self.pre_quant(gaussian)

        # embed = self.encoder(gaussian.permute(0, 2, 1))
        # pos_embed = self.sin_position_encoding.query(indice)
        # embed = self.pre_quant(embed.permute(0, 2, 1) + pos_embed)

        f_hat, indices, commit_loss = self.residual_vq(
            embed, freeze_codebook = False if not self.training else True
        )

        output = self.decoder(self.post_quant(f_hat).permute(0, 2, 1))
        output = output.permute(0, 2, 1)

        B, S, C = output.shape
        output = output.reshape(B, S, 2, C//2)

        return output, indices, commit_loss
    # ===================================================================================

    def gaussian2idx(self, gaussian, indice=None):
        """
        Convert two Gaussian into idx
        """
        gaussian = self.scale_gs.forward(gaussian)
        gaussian = self.sin_encoder(gaussian)
        gaussian = gaussian.flatten(-2, -1)

        embed = self.encoder(gaussian.permute(0, 2, 1))
        embed = self.pre_quant(embed.permute(0, 2, 1))

        f_hat, indices, _ = self.residual_vq(embed, freeze_codebook = False)
        return f_hat, indices
    
    def idx2gaussian(self, f_hat: torch.Tensor, indice: torch.Tensor):
        """
        Convert idx into two Gaussian
        """
        pos_embed = self.sin_position_encoding.query(indice)

        output = self.decoder((self.post_quant(f_hat)+pos_embed).permute(0, 2, 1))
        output = output.permute(0, 2, 1)

        B, S, C = output.shape
        output = output.reshape(B, S, 2, C//2)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), amsgrad=True, lr=0.0001, weight_decay=0.1)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineWarmupScheduler(
                    optimizer, 
                    self.trainer.max_epochs//10, 
                    self.trainer.max_epochs,
                ),
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
