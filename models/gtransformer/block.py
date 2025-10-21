import torch
import torch.nn as nn
try:
    import flash_attn
    use_flash_attn = True
    if flash_attn.__version__[0] < '2':
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func as flash_attn_varlen_func
    elif flash_attn.__version__[0] < '3':
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
except:
    use_flash_attn = False
import torch.nn.functional as F


class RoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.freq_scale = nn.Parameter(torch.tensor([0.0]))

    def forward(self, coord, embed):
        """
        coord: [..., 3]
        embed: [..., 6*C]
        """
        assert coord.shape[-1] == 3
        assert embed.shape[-1] % 6 == 0
        
        x, y, z = coord.chunk(3, dim=-1)
        
        x = x * self.freq_scale.exp().clip(max=1e3)
        y = y * self.freq_scale.exp().clip(max=1e3)
        z = z * self.freq_scale.exp().clip(max=1e3)
        
        e1, e2, e3, e4, e5, e6 = embed.chunk(6, dim=-1)

        o0 = e1*torch.cos(x)-e2*torch.sin(x)
        o1 = e2*torch.cos(x)+e1*torch.sin(x)
        o2 = e3*torch.cos(y)-e4*torch.sin(y)
        o3 = e4*torch.cos(y)+e3*torch.sin(y)
        o4 = e5*torch.cos(z)-e6*torch.sin(z)
        o5 = e6*torch.cos(z)+e5*torch.sin(z)

        return torch.cat([o0, o1, o2, o3, o4, o5], dim=-1)

class SelfAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_heads = embed_dim // num_heads
        self.dropout   = dropout

        self.ln_prev   = nn.LayerNorm(embed_dim)
        self.qkv       = nn.Linear(embed_dim, self.embed_dim*3)
        self.rope      = RoPE()

        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, feat, pos, cu_seqlens_gs=None, mask=None):
        """
        feat            : [N, C] or [B, L, C]   feature
        pos             : [N, 3] or [B, L, 3]   position
        cu_seqlens_gs   : [B+1] or None         cumulative sequence lengths
        mask            : [B, 1, L, L] or None  attention mask
        
        return          : [N, C] or [B, L, C]   output
        """

        x       = self.ln_prev(feat)   # [N,  C] or [B, L,  C]
        qkv     = self.qkv(x)          # [N, 3C] or [B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1) # [N,  C] or [B, L,  C]

        attn_drop = self.dropout if self.training else 0.0

        if use_flash_attn:
            assert (cu_seqlens_gs is not None) and (len(feat.shape) == 2)

            (N, C), H, D = feat.shape, self.num_heads, self.dim_heads

            q = q.view(N, H, D)
            k = k.view(N, H, D)
            v = v.view(N, H, D)

            # breakpoint()
            pos = pos.unsqueeze(1).expand(-1, H, -1)
            q = self.rope(pos, q) # [N, H, D]
            k = self.rope(pos, k) # [N, H, D]

            max_seqlen_qk = torch.diff(cu_seqlens_gs).max().item()
            
            out = flash_attn_varlen_func(
                q.half(), k.half(), v.half(), 
                cu_seqlens_gs, cu_seqlens_gs,
                max_seqlen_qk, max_seqlen_qk, 
                attn_drop, 
                return_attn_probs=False, causal=False
            )      # [N, H, D]
            
            out = out.reshape(N, C)              # [N, C]
        else:
            assert (mask is not None) and (len(feat.shape) == 3)

            (B, L, C), H, D = feat.shape, self.num_heads, self.dim_heads

            q = q.view(B, L, H, D) # [B, L, H, D]
            k = k.view(B, L, H, D) # [B, L, H, D]
            v = v.view(B, L, H, D) # [B, L, H, D]
            
            q = q.permute(0, 2, 1, 3) # [B, H, L, D]
            k = k.permute(0, 2, 1, 3) # [B, H, L, D]
            v = v.permute(0, 2, 1, 3) # [B, H, L, D]

            pos = pos.unsqueeze(1).expand(-1, H, -1, -1)
            q = self.rope(pos, q) # [B, H, L, D]
            k = self.rope(pos, k) # [B, H, L, D]

            out = F.scaled_dot_product_attention(q, k, v, mask, attn_drop) # [B, H, L, D]
            out = out.permute(0, 2, 1, 3)                                  # [B, L, H, D]
            out = out.reshape(B, L, C)
        
        return self.proj_drop(self.proj(out))

class CrossAttn(nn.Module):
    def __init__(self, embed_dim, kv_dims, num_heads, dropout=0.1) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_heads = embed_dim // num_heads
        self.dropout   = dropout

        self.ln_q    = nn.LayerNorm(embed_dim)
        self.q_proj  = nn.Linear(embed_dim,  embed_dim)
        self.kv_proj = nn.Linear(kv_dims, embed_dim*2)

        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, feat, kv_packed, cu_seqlens_gs=None, cu_seqlens_kv=None, mask=None):
        """
        feat            : [N, C] or [B, L, C]   feature
        kv_packed       : [M, C] or [B, S, C]   key-value packed feature
        cu_seqlens_gs   : [B+1] or None         cumulative sequence lengths
        cu_seqlens_kv   : [M+1] or None         cumulative sequence lengths
        mask            : [B, 1, L, S] or None  attention mask

        return          : [N, C] or [B, L, C]   output
        """
        q         = self.q_proj(self.ln_q(feat))        # [N,  C] or [B, L,  C]
        kv        = self.kv_proj(kv_packed)             # [M, 2C] or [B, S, 2C]
        k, v      = kv.chunk(2, dim=-1)                 # [M,  C] or [B, S,  C]

        attn_drop = self.dropout if self.training else 0.0

        if use_flash_attn:
            assert (cu_seqlens_gs is not None) and (len(feat.shape) == 2) and (len(kv_packed.shape) == 2)

            (N, C), M, H, D = feat.shape, kv_packed.shape[0], self.num_heads, self.dim_heads

            q = q.view(N, H, D)
            k = k.view(M, H, D)
            v = v.view(M, H, D)

            max_seqlen_q = torch.diff(cu_seqlens_gs).max().item()
            max_seqlen_k = torch.diff(cu_seqlens_kv).max().item()

            out = flash_attn_varlen_func(
                q.half(), k.half(), v.half(), 
                cu_seqlens_gs, cu_seqlens_kv, 
                max_seqlen_q, max_seqlen_k, attn_drop,
                return_attn_probs=False, causal=False,
            )

            out = out.reshape(N, C)
        else:
            assert (mask is not None) and (len(feat.shape) == 3) and (len(kv_packed.shape) == 3)

            (B, L, C), S, H, D = feat.shape, kv_packed.shape[1], self.num_heads, self.dim_heads

            q = q.view(B, L, H, D)
            k = k.view(B, S, H, D)
            v = v.view(B, S, H, D)

            q = q.permute(0, 2, 1, 3) # [B, H, L, D]
            k = k.permute(0, 2, 1, 3) # [B, H, S, D]
            v = v.permute(0, 2, 1, 3) # [B, H, S, D]

            out = F.scaled_dot_product_attention(q, k, v, mask, attn_drop) # [B, H, L, D]
            out = out.permute(0, 2, 1, 3)                                  # [B, L, H, D]
            out = out.reshape(B, L, C)                                     # [B, L, C]

        return self.proj_drop(self.proj(out))


class FFN(nn.Module):
    def __init__(self, embed_dim, dropout=0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ln_ffn  = nn.LayerNorm(self.embed_dim)
        self.ffn     = nn.Sequential(
            nn.Linear(self.embed_dim, 2*self.embed_dim),
            nn.GELU(),
            nn.Linear(2*self.embed_dim, self.embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ln_ffn(x)
        y = self.dropout(self.ffn(y))
        return y

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.sa  = SelfAttn( embed_dim, num_heads, dropout)
        self.ca  = CrossAttn(embed_dim, embed_dim, num_heads, dropout)
        self.ffn = FFN(embed_dim, dropout)
    
    def forward(self, feat, pos, kv_packed, cu_seqlens_gs=None, cu_seqlens_kv=None, mask=None):
        feat = feat+self.sa(feat, pos, cu_seqlens_gs, mask)
        feat = feat+self.ca(feat, kv_packed, cu_seqlens_gs, cu_seqlens_kv, mask)
        feat = feat+self.ffn(feat)
        return feat

if __name__ == '__main__':
    rope = RoPE()
    x = torch.randn(10, 4, 8, 3) # N, H, L, C
    y = torch.randn(10, 4, 8, 6*8)
    z = rope(x, y)

    x = torch.randn(10, 8, 64)

    selfattn = SelfAttn(64, 4)
    z = selfattn(x)
    
    q = torch.randn(10, 8, 1536)
    kv = torch.randn(10, 6, 1536)

    crossattn = CrossAttn(1536, 1536, 4)
    z = crossattn(q, kv)
    breakpoint()
