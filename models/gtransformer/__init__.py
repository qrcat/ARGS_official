from .block import SelfAttn, CrossAttn, FFN, Block
import torch
import torch.nn as nn


class GTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, num_heads, dropout):
        super(GTransformer, self).__init__()
        self.embedding_dim = embedding_dim

        self.input_proj = nn.Sequential(nn.Linear(input_dim+1, embedding_dim),)
        
        self.uncond     = nn.Embedding(256, embedding_dim)
        self.ln_cond    = nn.LayerNorm(embedding_dim)

        self.blocks     = nn.ModuleList([Block(embedding_dim, num_heads, dropout) for _ in range(num_layers)])

        self.split_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )
        self.dense_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 2*14)
        )

    def forward(self, feat, pos, cond, cu_seqlens_gs=None, cu_seqlens_kv=None, mask=None, split_mask=None):
        assert (len(feat.shape)==2 and len(cond.shape)==2 and cu_seqlens_gs is not None and cu_seqlens_kv is not None) or (len(feat.shape)==3 and len(cond.shape)==3 and mask is not None)

        if len(feat.shape) == 2:
            for i in range(cu_seqlens_kv.shape[0]-1):
                # if p < 0.5:
                cond[cu_seqlens_kv[i]:cu_seqlens_kv[i+1]] = self.uncond.weight[:cu_seqlens_kv[i+1]-cu_seqlens_kv[i]]
        else:
            # B, L, S
            ...

        # add more feature
        volume = torch.prod(feat[..., 7:10], dim=-1, keepdims=True)
        feat   = torch.cat([feat, volume], dim=-1)
        feat   = self.input_proj(feat)

        cond   = self.ln_cond(cond)

        for block in self.blocks:
            feat = block(feat, pos, cond, cu_seqlens_gs, cu_seqlens_kv, mask)

        split  = self.split_head(feat)
        
        if split_mask is not None:
            dense = self.dense_head(feat[split_mask])
        else:
            dense = self.dense_head(feat)
        
        dense  = dense.view(-1, 2, 14)
        # dense[..., 3:4] = torch.sigmoid(dense[..., 3:4])

        return split, dense
