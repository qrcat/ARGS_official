from .block import SelfAttn, CrossAttn, FFN, Block
from lightning import LightningModule
from torch.amp import autocast
from typing import Optional, Union
import torch
import torch.nn as nn


class GTransformer(LightningModule):
    def __init__(self, input_dim, cond_dim, embedding_dim, num_layers, num_heads, dropout):
        super(GTransformer, self).__init__()
        self.embedding_dim = embedding_dim

        self.proj_g     = nn.Linear(input_dim, embedding_dim)
        self.proj_f     = nn.Linear(cond_dim, embedding_dim)
        
        self.ln_f       = nn.LayerNorm(embedding_dim)

        self.blocks     = nn.ModuleList([Block(embedding_dim, num_heads, dropout) for _ in range(num_layers)])

        self.split_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.dense_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 2*14)
        )

    def forward(
            self, 
            gs_params: torch.Tensor,
            positions: torch.Tensor, 
            condition: torch.Tensor, 
            cu_seqlens_gs_params: torch.Tensor=None,
            cu_seqlens_condition: torch.Tensor=None,
            max_seqlen_gs_params: Union[int]=None,
            max_seqlen_condition: Union[int]=None,
            gs_params_attn_mask: torch.Tensor=None,
            condition_attn_mask: torch.Tensor=None,
            split_mask: Optional[torch.Tensor]=None
        ):
        if len(gs_params.shape)==2 and len(condition.shape)==2:
            assert (cu_seqlens_gs_params is not None and 
                    cu_seqlens_condition is not None and 
                    max_seqlen_gs_params is not None and 
                    max_seqlen_condition is not None and 
                    gs_params_attn_mask is None and 
                    condition_attn_mask is None)
        elif len(gs_params.shape)==3 and len(condition.shape)==3:
            assert (cu_seqlens_gs_params is None and 
                    cu_seqlens_condition is None and 
                    max_seqlen_gs_params is None and 
                    max_seqlen_condition is None and
                    gs_params_attn_mask is not None and
                    condition_attn_mask is not None)
        else:
            raise ValueError(f"Invalid input shape: {gs_params.shape} and {condition.shape}")

        condition      = self.proj_f(condition)
        condition      = self.ln_f(condition)

        with autocast('cuda', dtype=torch.float32):
            feat = self.proj_g(gs_params)

            for block in self.blocks:
                feat = block(
                    feat, positions, condition, 
                    cu_seqlens_gs_params, cu_seqlens_condition, max_seqlen_gs_params, max_seqlen_condition, 
                    gs_params_attn_mask, condition_attn_mask
                )

            split  = self.split_head(feat)

            if split_mask is not None:
                dense = self.dense_head(feat[split_mask])
            else:
                dense = self.dense_head(feat)

            dense     = dense.view(-1, 2, 14)

        return split, dense


class MaskedTransformer(GTransformer):
    def __init__(self, input_dim, cond_dim, embedding_dim, num_layers, num_heads, dropout):
        super().__init__(input_dim, cond_dim, embedding_dim, num_layers, num_heads, dropout)

        self.f_uncond        = nn.Embedding(1, embedding_dim) # this is used for unconditional training
        self.dense_head[-1]  = nn.Linear(embedding_dim, input_dim)

    def forward(
            self, 
            gs_params: torch.Tensor,
            positions: torch.Tensor, 
            condition: torch.Tensor, 
            cu_seqlens_gs_params: torch.Tensor=None,
            cu_seqlens_condition: torch.Tensor=None,
            max_seqlen_gs_params: Union[int]=None,
            max_seqlen_condition: Union[int]=None,
            gs_params_attn_mask: torch.Tensor=None,
            condition_attn_mask: torch.Tensor=None,
            split_mask: Optional[torch.Tensor]=None
        ):
        if len(gs_params.shape)==2 and len(condition.shape)==2:
            assert (cu_seqlens_gs_params is not None and 
                    cu_seqlens_condition is not None and 
                    max_seqlen_gs_params is not None and 
                    max_seqlen_condition is not None and 
                    gs_params_attn_mask is None and 
                    condition_attn_mask is None)
        elif len(gs_params.shape)==3 and len(condition.shape)==3:
            assert (cu_seqlens_gs_params is None and 
                    cu_seqlens_condition is None and 
                    max_seqlen_gs_params is None and 
                    max_seqlen_condition is None and
                    gs_params_attn_mask is not None and
                    condition_attn_mask is not None)
        else:
            raise ValueError(f"Invalid input shape: {gs_params.shape} and {condition.shape}")
        
        condition     = self.proj_f(condition)      # [M, D] or [B, S, D] -> [M, C] or [B, S, C]
        condition     = self.ln_f(condition)        # [M, C] or [B, S, C]
        
        with autocast('cuda', dtype=torch.float32):
            feat      = self.proj_g(gs_params) # [N, C] or [B, L, C]
            # step 1: masking
            if split_mask is not None:
                feat[split_mask] = self.f_uncond.weight.expand(split_mask.sum(), -1).to(feat.dtype)
            
            for block in self.blocks:
                feat = block(
                    feat, positions, condition, 
                    cu_seqlens_gs_params, cu_seqlens_condition, max_seqlen_gs_params, max_seqlen_condition, 
                    gs_params_attn_mask, condition_attn_mask
                )
            split  = self.split_head(feat)

            if split_mask is not None:
                dense = self.dense_head(feat[split_mask])
            else:
                dense = self.dense_head(feat)

        return split, dense
