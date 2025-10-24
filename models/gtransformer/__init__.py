from .block import SelfAttn, CrossAttn, FFN, Block
import torch
import torch.nn as nn


class GTransformer(nn.Module):
    def __init__(self, input_dim, cond_dim, embedding_dim, num_layers, num_heads, dropout):
        super(GTransformer, self).__init__()
        self.embedding_dim = embedding_dim

        self.proj_g     = nn.Linear(input_dim, embedding_dim)
        self.proj_f     = nn.Linear(cond_dim, embedding_dim)
        
        # self.f_uncond   = nn.Embedding(256, cond_dim) # this is used for unconditional training
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

    def forward(self, gs_params, pos, cond, cu_seqlens_gs=None, cu_seqlens_kv=None, max_seqlen_gs=None, max_seqlen_kv=None, mask=None, split_mask=None):
        assert (len(gs_params.shape)==2 and len(cond.shape)==2 and cu_seqlens_gs is not None and cu_seqlens_kv is not None) or (len(gs_params.shape)==3 and len(cond.shape)==3 and mask is not None)

        # if len(feat.shape) == 2:
        #     for i in range(cu_seqlens_kv.shape[0]-1):
        #         if torch.rand(1) < 0.3:
        #             cond[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]] = self.f_uncond.weight[:cu_seqlens_kv[i+1]-cu_seqlens_kv[i]]
        # else:
        #     # B, L, S
        #     ...
        cond      = self.proj_f(cond)
        cond      = self.ln_f(cond)

        gs_params = self.proj_g(gs_params)

        
        for block in self.blocks:
            gs_params = block(gs_params, pos, cond, cu_seqlens_gs, cu_seqlens_kv, max_seqlen_gs, max_seqlen_kv, mask)

        split  = self.split_head(gs_params)
        
        if split_mask is not None:
            dense = self.dense_head(gs_params[split_mask])
        else:
            dense = self.dense_head(gs_params)

        dense     = dense.view(-1, 2, 14)
        # activate
        dense[...,  3:4] = torch.nn.functional.softplus(dense[..., 3:4])

        return split, dense
