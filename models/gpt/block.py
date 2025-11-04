import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
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

    def get_freqs(self, dim, base=10000):
        half = dim // 2
        freqs = base ** (-torch.arange(0, half, 1, dtype=torch.float32) / half)
        return freqs  # [half]

    def forward(self, coord, embed, hdim=-2):
        """
        coord: [..., 3]
        embed: [..., H,...]
        """
        assert coord.shape[-1] == 3
        assert embed.shape[hdim] % 3 == 0
        
        x, y, z = coord.chunk(3, dim=-1)
        
        freqs = self.get_freqs(embed.shape[-1]).to(x.device, x.dtype)
        
        x = x * freqs
        y = y * freqs
        z = z * freqs

        f_x, f_y, f_z = embed.chunk(3, dim=hdim)

        e1, e2 = f_x.chunk(2, dim=-1)
        e3, e4 = f_y.chunk(2, dim=-1)
        e5, e6 = f_z.chunk(2, dim=-1)

        o0 = e1*torch.cos(x)-e2*torch.sin(x)
        o1 = e2*torch.cos(x)+e1*torch.sin(x)
        o2 = e3*torch.cos(y)-e4*torch.sin(y)
        o3 = e4*torch.cos(y)+e3*torch.sin(y)
        o4 = e5*torch.cos(z)-e6*torch.sin(z)
        o5 = e6*torch.cos(z)+e5*torch.sin(z)
        
        return torch.cat([torch.cat([o0, o1], dim=-1),torch.cat([o2, o3], dim=-1),torch.cat([o4, o5], dim=-1),], dim=hdim)

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

    def forward(self, feat, pos, block_mask=None, past_kv=None, use_cache=False):
        """
        feat            : [N, C] or [B, L, C]   feature
        pos             : [N, 3] or [B, L, 3]   position
        cu_seqlens_gs   : [B+1] or None         cumulative sequence lengths
        mask            : [B, 1, L, L] or None  attention mask
        
        return          : [N, C] or [B, L, C]   output
        """
        x       = self.ln_prev(feat)                    # [N,  C] or [B, L,  C]
        qkv     = self.qkv(x)                           # [N, 3C] or [B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1)                  # [N,  C] or [B, L,  C]
        
        (B, L, C), H, D = feat.shape, self.num_heads, self.dim_heads

        q = q.view(B, L, H, D)                      # [B, L, H, D]
        k = k.view(B, L, H, D)                      # [B, L, H, D]
        v = v.view(B, L, H, D)                      # [B, L, H, D]
        
        q = q.permute(0, 2, 1, 3)                   # [B, H, L, D]
        k = k.permute(0, 2, 1, 3)                   # [B, H, L, D]
        v = v.permute(0, 2, 1, 3)                   # [B, H, L, D]

        pos = pos.unsqueeze(1)
        q = self.rope(pos.half(), q, hdim=-3)              # [B, H, L, D]
        k = self.rope(pos.half(), k, hdim=-3)              # [B, H, L, D]
        
        if past_kv is not None:
            past_k, past_v = past_kv
            
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        out = flex_attention(q, k, v, block_mask=block_mask)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        out = self.proj_drop(self.proj(out))

        new_kv = (k, v) if use_cache else None
        return (out, new_kv)

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
        x = self.ln_ffn(x)
        x = self.dropout(self.ffn(x))
        return x

class SABlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1) -> None:
        super().__init__()
        self.sa  = SelfAttn( embed_dim, num_heads, dropout)
        self.ffn = FFN(embed_dim, dropout)
    
    def forward(self, feat, pos, block_mask=None, past_kv=None, use_cache=False):
        out, new_kv = self.sa(feat, pos, block_mask, past_kv, use_cache)
        feat = feat + out
        feat = feat + self.ffn(feat)
        return feat, new_kv
