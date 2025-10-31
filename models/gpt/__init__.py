from .block import SABlock
from lightning import LightningModule
from torch.amp import autocast
from typing import Optional, Union
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask


class GPT(LightningModule):
    def __init__(self, input_dim, embedding_dim, output_dim, num_layers=1, num_heads=3, vocal_dim=64, dropout=0.0,):
        super(GPT, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(256*5, vocal_dim)

        self.proj_x    = nn.Linear(vocal_dim*3, embedding_dim)
        self.proj_o    = nn.Linear(vocal_dim*1, embedding_dim)
        self.proj_f    = nn.Linear(vocal_dim*3, embedding_dim)
        self.proj_s    = nn.Linear(vocal_dim*3, embedding_dim)
        self.proj_q    = nn.Linear(vocal_dim*4, embedding_dim)

        self.blocks     = nn.ModuleList([SABlock(embedding_dim, num_heads, dropout) for _ in range(num_layers)])

        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1+output_dim)
        )

        for block in self.blocks:
            block.sa.proj.weight.data.mul_(0.01)
            block.ffn.ffn[-1].weight.data.mul_(0.01)

    def forward(
            self, 
            sequence,
            position,
            mask_value,
        ):
        with autocast('cuda', dtype=torch.float16):
            embed_x = self.embedding(sequence[..., :3]).flatten(-2)
            embed_o = self.embedding(sequence[..., 3:4]).flatten(-2)
            embed_f = self.embedding(sequence[..., 4:7]).flatten(-2)
            embed_s = self.embedding(sequence[..., 7:10]).flatten(-2)
            embed_q = self.embedding(sequence[..., 10:]).flatten(-2)

            embed_x = self.proj_x(embed_x)
            embed_o = self.proj_o(embed_o)
            embed_f = self.proj_f(embed_f)
            embed_s = self.proj_s(embed_s)
            embed_q = self.proj_q(embed_q)

            feat = embed_x + embed_o + embed_f + embed_s + embed_q

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= mask_value[b, kv_idx]
            
            B, S, C = feat.shape

            block_mask = create_block_mask(causal_mask, B=B, H=self.hparams.num_heads, Q_LEN=S, KV_LEN=S, BLOCK_SIZE=1, device="cuda")

            for block in self.blocks:
                feat = block(feat, position, block_mask)

            return self.output_head(feat)
