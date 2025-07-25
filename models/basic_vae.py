import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder',]


class SinPositionEncoding(L.LightningModule):
    def __init__(self, d_model, base=10000):
        super().__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, max_sequence_length):
        pe = torch.zeros(max_sequence_length, self.d_model, dtype=torch.float)  # size(max_sequence_length, d_model)
        exp_1 = torch.arange(self.d_model // 2, dtype=torch.float)  # 初始化一半维度，sin位置编码的维度被分为了两部分
        exp_value = exp_1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)  # size(dmodel/2)
        out = torch.arange(max_sequence_length, dtype=torch.float)[:, None] @ alpha[None, :]  # size(max_sequence_length, d_model/2)
        embedding_sin = torch.sin(out)
        embedding_cos = torch.cos(out)

        pe[:, 0::2] = embedding_sin  # 奇数位置设置为sin
        pe[:, 1::2] = embedding_cos  # 偶数位置设置为cos
        return pe


class Encoder(L.LightningModule):
    def __init__(
        self, *, in_channels=14, seq=512, z_channels=512,
        nhead=(8, 8), nlayer=(3, 3),
        dropout=0.0, 
    ):
        super().__init__()
        self.seq = seq
        self.in_channels = in_channels
        self.z_channels = z_channels

        # transfer input to ch channels        
        self.input_head = torch.nn.Linear(in_channels, z_channels)
        
        # local Transformer encoder
        self.local_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=z_channels, nhead=nhead[0], batch_first=True, dropout=dropout,
            ),
            num_layers=nlayer[0]
        )
        self.local_pooler = torch.nn.Sequential(
            torch.nn.MaxPool2d([seq, 1]), torch.nn.Flatten(-2, -1)
        )

        # global Transformer encoder
        self.global_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=z_channels, nhead=nhead[1], batch_first=True
            ),
            num_layers=nlayer[1],
        )

    def forward(self, x):
        global_batch, local_batch, seq, _ = x.shape
        
        input_3d = x.reshape(global_batch * local_batch, seq, -1)
        input_3d = self.input_head(input_3d)
        
        local_features = self.local_encoder(input_3d)
        local_features = self.local_pooler(local_features)
        local_features = local_features.view(global_batch, local_batch, -1)
        
        global_features = self.global_encoder(local_features)

        return global_features


class Decoder(L.LightningModule):
    def __init__(
        self, *, in_channels=14, seq=512, z_channels=512,
        nhead=(8, 8), nlayer=(3, 3),
        dropout=0.0,
    ):
        super().__init__()
        self.seq = seq
        self.in_channels = in_channels
        self.z_channels = z_channels
        
        # postion encoding
        self.position_encoder = SinPositionEncoding(d_model=z_channels, base=10000)

        # transformer decoder
        self.global_decoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=z_channels, nhead=nhead[0], batch_first=True, dropout=dropout,
            ),
            num_layers=nlayer[0],
        )

        self.local_decoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=z_channels, nhead=nhead[1], batch_first=True, dropout=dropout,
            ),
            num_layers=nlayer[1],
        )

        # output head
        self.output_head = torch.nn.Linear(z_channels, in_channels)
    
    def forward(self, z):
        global_batch, local_batch, _ = z.shape

        memory = self.global_decoder(z)
        memory = memory.view(global_batch*local_batch, 1, -1)
        
        dummy_token = self.position_encoder(self.seq)[None, ...].repeat(global_batch*local_batch, 1, 1)
        dummy_token = dummy_token.to(device=z.device, dtype=z.dtype)

        output = self.local_decoder(dummy_token+memory)
        output = output.view(global_batch, local_batch, self.seq, -1)
        
        return self.output_head(output)
