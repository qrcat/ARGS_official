from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F
import lightning as L
import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(L.LightningModule):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25, test: bool = False
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((self.vocab_size,), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        
        self.test = test
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'znorm={self.using_znorm}, beta={self.beta}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BWC: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BWC.dtype
        if dtype != torch.float32: f_BWC = f_BWC.float()
        B, W, C = f_BWC.shape
        f_no_grad = f_BWC.detach()

        with torch.amp.autocast('cuda', enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BWC.device)

            if self.using_znorm:
                f_NC = f_BWC.view(-1, C)
                idx_N = torch.argmax(f_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                f_NC = f_BWC.view(-1, C)
                d_no_grad = torch.sum(f_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(f_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1)
                
            hit_V = idx_N.bincount(minlength=self.vocab_size).float()
            if self.training:
                if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
            # calc loss
            idx_BW = idx_N.view(B, -1)
            f_hat = self.embedding(idx_BW)
            
            if self.training and dist.initialized():
                handler.wait()
                if self.record_hit == 0: self.ema_vocab_hit_SV.copy_(hit_V)
                elif self.record_hit < 100: self.ema_vocab_hit_SV.mul_(0.9).add_(hit_V.mul(0.1))
                else: self.ema_vocab_hit_SV.mul_(0.99).add_(hit_V.mul(0.01))
                self.record_hit += 1
            vocab_hit_V.add_(hit_V)
            mean_vq_loss += F.mse_loss(f_hat.data, f_BWC).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            f_hat = (f_hat.data - f_no_grad).add_(f_BWC)
        
        margin = tdist.get_world_size() * (f_BWC.numel() / f_BWC.shape[1]) / self.vocab_size * 0.08 if self.training and dist.initialized() else 0
        # margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV >= margin).float().mean().item() * 100]
        else: usages = None
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        # TODO: change it
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBW_or_fhat(self, f_BWC: torch.Tensor, to_fhat: bool) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, W, C = f_BWC.shape
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
                
        # find the nearest embedding
        f_NC = f_BWC.view(-1, C)
        if self.using_znorm:
            idx_N = torch.argmax(f_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
        else:
            d_no_grad = torch.sum(f_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(f_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)
        
        idx_BW = idx_N.view(B, W)

        f_hat = self.embedding(idx_BW)

        f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_BW)
    
        return f_hat_or_idx_Bl
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


