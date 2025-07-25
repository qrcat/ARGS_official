"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import lightning as L
from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


class VQVAE(L.LightningModule):
    def __init__(
        self, vocab_size=4096, seqs=[64, 128, 256, 512], z_channels=512, dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        test_mode=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml

        encoders = []
        decoders = []
        for seq in seqs:
            ddconfig = dict(
                in_channels=14, seq=seq, z_channels=z_channels,
                nhead=(8, 8), nlayer=(3, 3),
                dropout=dropout,
            )
            encoders.append(Encoder(**ddconfig))
            decoders.append(Decoder(**ddconfig))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        
        self.vocab_size = vocab_size
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            test=test_mode,
        )
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

        
    
    def training_step(self, batch, batch_idx):
        data_level = {
            'scale_0': 0,
            'scale_1': 1,
            'scale_2': 2,
            'scale_3': 3,
        }

        loss = 0.0

        for key, level in data_level.items():
            y = batch[key]

            x = batch[f"{key}_normalized"]

            std = batch[f"{key}_std"]
            mean = batch[f"{key}_mean"]

            pred, _, vq_loss = self.forward(x, level)
            pred = ((pred * std) + mean) # unnormalized

            loss_xyz = (torch.norm(pred[..., :3]-y[..., :3], dim=-1)).mean()
            loss_opacity = torch.nn.functional.l1_loss(
                torch.sigmoid(pred[..., 3:4]),
                torch.sigmoid(y[..., 3:4]),
            )
            loss_features = torch.nn.functional.l1_loss(pred[..., 4:7], y[..., 4:7])
            loss_scale = torch.nn.functional.l1_loss(
            torch.nn.functional.softplus(pred[..., 7:10]),
            torch.nn.functional.softplus(y[..., 7:10]),
            )
            loss_quat = torch.nn.functional.l1_loss(
                torch.nn.functional.normalize(pred[..., 10:], dim=-1), y[..., 10:]
            )

            loss += loss_xyz + loss_opacity + loss_features + 0.001*loss_scale + loss_quat + vq_loss

        return loss 

    def configure_optimizers(self):
        # 示例: 配置优化器
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, i, ret_usages=False):   # -> rec_B3HW, idx_N, loss
        f_hat, usages, vq_loss = self.quantize(self.encoders[i](inp), ret_usages=ret_usages)
        return self.decoders[i](f_hat), usages, vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_gaussian(self, f_hat: torch.Tensor):
        return self.decoder(f_hat)
    
    def gaussian_to_idxBl(self, inp_img_no_grad: torch.Tensor) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.encoder(inp_img_no_grad)
        return self.quantize.f_to_idxBW_or_fhat(f, to_fhat=False)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBW_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


