import torch.nn as nn
import torch


class SinPositionEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length, base=100000):
        super().__init__()
        self.d_model = d_model
        self.base = base

        # 预计算0~max_sequence_length-1的完整位置编码并注册为buffer
        position = torch.arange(max_sequence_length).unsqueeze(1)  # shape: (max_seq, 1)
        
        # 计算除数项: base^(2i/d_model)的倒数
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * 
            (-torch.log(torch.tensor(self.base)) / self.d_model)
        )  # shape: (d_model//2,)
        
        # 计算角度
        angles = position * div_term  # shape: (max_seq, d_model//2)
        
        # 初始化位置编码张量
        pe = torch.zeros(max_sequence_length, self.d_model)
        
        # 填充正弦和余弦值：偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(angles)  # 0, 2, 4,...维度
        pe[:, 1::2] = torch.cos(angles)  # 1, 3, 5,...维度

        self.register_buffer('pe', pe)  # 存储在buffer中，自动管理设备和模型保存

    def forward(self, x):
        return self.pe[None, :x.shape[1]] + x

    def query(self, indice):
        return self.pe[indice, :]

class SinEncoder(nn.Module):
    """
    Map (x,) to [sin(x), cos(x), sin(2x), cos(2x), ...]
    """
    def __init__(self, dmodel=32):
        super().__init__()
        self.dmodel = dmodel
        frequencies = torch.pi*2**torch.arange(0, dmodel//2, dtype=torch.float32)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x):
        x_scaled = x * self.frequencies  # (batch_size, ..., dims)
        sin_components = torch.sin(x_scaled)
        cos_components = torch.cos(x_scaled)
        return torch.cat([x, sin_components, cos_components], dim=-1)

class GaussianSinEncoder(nn.Module):
    def __init__(self, dmodel=32, embed_x=True, embed_o=False, embed_f=False, embed_s=True, embed_q=False):
        super().__init__()
        self.dmodel = dmodel
        self.out_dim = 0
        if embed_x:
            self.embed_x = SinEncoder(dmodel)
            self.out_dim += dmodel*3+3
        else:
            self.embed_x = nn.Identity()
            self.out_dim += 3
        if embed_o:
            self.embed_o = SinEncoder(dmodel)
            self.out_dim += dmodel+1
        else:
            self.embed_o = nn.Identity()
            self.out_dim += 1
        if embed_f:
            self.embed_f = SinEncoder(dmodel)
            self.out_dim += dmodel*3+3
        else:
            self.embed_f = nn.Identity()
            self.out_dim += 3
        if embed_s:
            self.embed_s = SinEncoder(dmodel)
            self.out_dim += dmodel*3+3
        else:
            self.embed_s = nn.Identity()
            self.out_dim += 3
        if embed_q:
            self.embed_q = SinEncoder(dmodel)
            self.out_dim += dmodel*4+4
        else:
            self.embed_q = nn.Identity()
            self.out_dim += 4

    def forward(self, x, o=None, f=None, s=None, q=None, dims=-1):
        if o is None and f is None and s is None and q is None:
            x, o, f, s, q = x.split([3, 1, 3, 3, 4], dim=dims)
        x = self.embed_x(x.unsqueeze(-1)).flatten(-2)
        o = self.embed_o(o.unsqueeze(-1)).flatten(-2)
        f = self.embed_f(f.unsqueeze(-1)).flatten(-2)
        s = self.embed_s(s.unsqueeze(-1)).flatten(-2)
        q = self.embed_q(q.unsqueeze(-1)).flatten(-2)
        return torch.concat([x, o, f, s, q], dim=-1)
