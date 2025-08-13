from utils.quaternion import standardize_quaternion
from utils.args import ARDecoder
from utils.io import unpack_gaussian_parameters, train_gs2activated_gs, activated_gs2gs, save_ply

from tqdm import tqdm

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
import lightning as L

import torch


class BatchNormLD(nn.Module):
    """
    Batch Normalization on the last dimension
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNormLD, self).__init__()
        self.num_features = num_features  # 最后一个维度的大小
        self.eps = eps
        self.momentum = momentum
        
        # learnable gamma and beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # track running mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x: (*, num_features)
        
        if self.training:
            # compute mean and variance at last dimension
            dims = tuple(range(x.dim() - 1)) 
            N = x.numel() // self.num_features
            
            mean = torch.einsum('...c->c', x) / N
            
            x_minus_mean = x - mean
            var = torch.einsum('...c,...c->c', x_minus_mean, x_minus_mean) / N
            
            # update running mean and variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
            self.num_batches_tracked += 1
        else:
            # use running mean and variance during inference
            mean = self.running_mean
            var = self.running_var
        
        # normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # apply scale and shift
        out = self.gamma * x_normalized + self.beta
        
        return out

class BasicBlock(nn.Module):
    def __init__(self, layer_dims, drop_out=0.5):
        super().__init__()
        self.layer_dims = layer_dims
        self.drop_out = drop_out

        self.relu = nn.LeakyReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Linear(layer_dims, layer_dims),
            BatchNormLD(layer_dims),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),
            
            nn.Linear(layer_dims, layer_dims),
            BatchNormLD(layer_dims),
        )
    
    def forward(self, x):
        return self.relu(x + self.layers(x))

class Bottleneck(nn.Module):
    def __init__(self, layer_dims, drop_out=0.5):
        super().__init__()
        self.layer_dims = layer_dims
        self.drop_out = drop_out

        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            nn.Linear(layer_dims, layer_dims),
            BatchNormLD(layer_dims),
            nn.ReLU(),
            nn.Dropout(drop_out),
            
            nn.Linear(layer_dims, layer_dims),
            BatchNormLD(layer_dims),
            nn.ReLU(),
            nn.Dropout(drop_out),
            
            nn.Linear(layer_dims, layer_dims),
            BatchNormLD(layer_dims),
        )

    def forward(self, x):
        return self.relu(x + self.layers(x))

class ResNet18(nn.Module):
    def __init__(self, input_dims=14, z_channels=512, output_dims=14, drop_out=None):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dims, z_channels),
            BatchNormLD(z_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            BasicBlock(z_channels, drop_out[0]),
            BasicBlock(z_channels, drop_out[0]),
            BasicBlock(z_channels, drop_out[1]),
            BasicBlock(z_channels, drop_out[1]),
            BasicBlock(z_channels, drop_out[2]),
            BasicBlock(z_channels, drop_out[2]),
            BasicBlock(z_channels, drop_out[3]),
            BasicBlock(z_channels, drop_out[3]),
        )

        self.output = nn.Sequential(
            nn.Linear(z_channels, output_dims),
        )
    
    def forward(self, x):
        return self.output(self.bottleneck(self.input(x)))

class ResNet18Encoder(ResNet18):
    def __init__(self, input_dims=14, z_channels=512, drop_out=[0.0, 0.0, 0.0, 0.0]):
        super().__init__(input_dims=input_dims, z_channels=z_channels, output_dims=z_channels, drop_out=drop_out)

class ResNet18Decoder(ResNet18):
    def __init__(self, z_channels=512, output_dims=14, drop_out=[0.0, 0.0, 0.0, 0.0]):
        super().__init__(input_dims=z_channels, z_channels=z_channels, output_dims=output_dims, drop_out=drop_out)

class MLPEncoder(nn.Module):
    def __init__(self, input_dims=14, z_channels=512, drop_out=[0.1, 0.2, 0.3], double_z=False):
        super().__init__()

        self.encoder = nn.Sequential(
            # process input to features space
            nn.Linear(input_dims, z_channels),
            nn.LeakyReLU(),
            nn.Dropout(drop_out[0]),
            BatchNormLD(z_channels),
            # backbone
            nn.Linear(z_channels, z_channels),
            nn.LeakyReLU(),
            nn.Dropout(drop_out[1]),
            BatchNormLD(z_channels),
            nn.Linear(z_channels, z_channels),
            nn.LeakyReLU(),
            nn.Dropout(drop_out[2]),
            BatchNormLD(z_channels),
            # features output
            nn.Linear(z_channels, z_channels if not double_z else z_channels*2),
        )

    def forward(self, gaussian):
        return self.encoder(gaussian)

class MLPDecoder(nn.Module):
    def __init__(self, z_channels=512, output_dims=14, drop_out=[0.1, 0.2, 0.3]):
        super().__init__()
        self.decoder = nn.Sequential(
            # process features to output space
            nn.Linear(z_channels, z_channels),
            nn.LeakyReLU(),
            nn.Dropout(drop_out[0]),
            BatchNormLastDim(z_channels),
            # backbone
            nn.Linear(z_channels, z_channels),
            nn.LeakyReLU(),
            nn.Dropout(drop_out[1]),
            BatchNormLastDim(z_channels),
            nn.Linear(z_channels, z_channels),
            nn.LeakyReLU(),
            nn.Dropout(drop_out[2]),
            BatchNormLastDim(z_channels),
            # output
            nn.Linear(z_channels, output_dims),
        )

    def forward(self, embedding):
        return self.decoder(embedding)

class SVAE(L.LightningModule):
    def __init__(
        self, 
        gaussian_dim=14, sin_dims=128, z_channels=512,
        # loss weight
        weight_x=10.0, 
        weight_o=1.0, 
        weight_f=1.0, 
        weight_s=10.0, 
        weight_q=1.0,
        kl_weight=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # use sin encode to embed the gaussian
        self.sin_encoder = SinEncoder(sin_dims)
        # scale gaussian into [-1, 1]
        self.scale_gs = ScaleGaussianSplat()
        # encoder-decoder
        self.encoder = MLPEncoder(gaussian_dim*sin_dims, z_channels, double_z=False)
        self.decoder = MLPDecoder(z_channels, 14)
        
        self.position_encoder = SinPositionEncoding(z_channels, max_sequence_length=65536)

        self.gs0_avg = nn.Linear(z_channels, z_channels)
        self.gs1_avg = nn.Linear(z_channels, z_channels)
        self.gs0_std = nn.Linear(z_channels, z_channels)
        self.gs1_std = nn.Linear(z_channels, z_channels)

    def reconstruct_loss(self, x, y):
        # compute target
        true_x, true_o, true_f, true_s, true_q = y.split([3, 1, 3, 3, 4], dim=-1)
        pred_x, pred_o, pred_f, pred_s, pred_q = x.split([3, 1, 3, 3, 4], dim=-1)

        loss_x = self.hparams.weight_x * (torch.norm(pred_x-true_x, dim=-1)).mean()

        loss_o = self.hparams.weight_o * torch.nn.functional.l1_loss(
            torch.sigmoid(pred_o),
            true_o
        )

        loss_f = self.hparams.weight_f * torch.nn.functional.l1_loss(pred_f, true_f)

        loss_s = self.hparams.weight_s * torch.nn.functional.l1_loss(torch.nn.functional.softplus(pred_s), true_s)

        loss_q = self.hparams.weight_q * (1-(torch.sum(torch.nn.functional.normalize(pred_q, dim=-1) * true_q, dim=-1).abs()).mean())

        loss =  loss_x + loss_o + loss_f + loss_s + loss_q

        return loss, (loss_x.item(), loss_o.item(), loss_f.item(), loss_s.item(), loss_q.item())

    def reparameterize(self, mu, logvar):
        """z = mu + eps * sigma"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ==================== forward is only used for training ====================
    def forward(self, target, indice):
        target = self.scale_gs.forward(target)
        target = self.sin_encoder(target.unsqueeze(-1)).flatten(-2, -1)
        token = self.encoder(target)

        gs0_avg = self.gs0_avg(token)
        gs1_avg = self.gs1_avg(token)
        gs0_std = self.gs0_std(token)
        gs1_std = self.gs1_std(token)

        gs0_embed = self.reparameterize(gs0_avg, gs0_std)
        gs1_embed = self.reparameterize(gs1_avg, gs1_std)

        gs0 = self.decoder(gs0_embed)
        gs1 = self.decoder(gs1_embed)

        loss_kl0 = self.kl_divergence(gs0_avg, gs0_std)
        loss_kl1 = self.kl_divergence(gs1_avg, gs1_std)

        return torch.stack([gs0, gs1], dim=-1), loss_kl0 + loss_kl1

    def kl_divergence(self, mu, logvar):
        """KL(q(z|x) || p(z))"""
        return -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    # ===========================================================================
    
    def encode(self, target, indice=None):
        if indice is None:
            indice = torch.arange(target.shape[1], device=target.device)[None]

        target = self.scale_gs.forward(target)
        target = self.sin_encoder(target.unsqueeze(-1)).flatten(-2, -1)
        token = self.encoder(target)

        return token

    def decode(self, token):
        gs0_avg = self.gs0_avg(token)
        gs1_avg = self.gs1_avg(token)
        gs0_std = self.gs0_std(token)
        gs1_std = self.gs1_std(token)

        gs0_embed = self.reparameterize(gs0_avg, gs0_std)
        gs1_embed = self.reparameterize(gs1_avg, gs1_std)

        gs0 = self.decoder(gs0_embed)
        gs1 = self.decoder(gs1_embed)

        return torch.stack([gs0, gs1], dim=-1)

    def training_step(self, batch, batch_idx):
        source = batch[0]
        target = batch[1]
        indice = batch[2]

        pred_s, loss_kl = self.forward(target, indice)

        loss_rc, loss_disp = self.reconstruct_loss(pred_s, source)
        
        loss = loss_rc + self.hparams.kl_weight * loss_kl / 2.0

        self.log_dict({
            "loss": loss.item(),
            "kl/loss": loss_kl.item(),
            "recon_source/loss": loss_rc.item(),
            "recon_source/x": loss_disp[0],
            "recon_source/o": loss_disp[1],
            "recon_source/f": loss_disp[2],
            "recon_source/s": loss_disp[3],
            "recon_source/q": loss_disp[4],
        })

        return loss

    def test_step(self, batch, batch_idx):
        batch
        for source, target, indice in zip(batch[0], batch[1], batch[2]):
            gaussian, _ = self.forward(target, indice)
            gaussian = gaussian.permute(0, 2, 1)
            gaussian = train_gs2activated_gs(gaussian)
            
            ardecoder = ARDecoder(gaussian.shape[0]*2, device=self.device)
            for i, gs in enumerate(tqdm(gaussian)):
                ardecoder.add(gs)
                if ardecoder.size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
                    ardecoder.save_ply(f'decode-{ardecoder.size}.ply')
            ardecoder.save_ply(f'decode-{ardecoder.size}.ply')


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)


class VAEGAR(L.LightningModule):
    def __init__(
        self, 
        gaussian_dim=14, sin_dims=128, z_channels=512,
        # loss weight
        weight_x=10.0, 
        weight_o=1.0, 
        weight_f=1.0, 
        weight_s=100.0, 
        weight_q=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # use sin encode to embed the gaussian
        self.sin_encoder = SinEncoder(sin_dims)
        # scale gaussian into [-1, 1]
        self.scale_gs = ScaleGaussianSplat()
        # encoder-decoder
        self.encoder = MLPEncoder(gaussian_dim*sin_dims, z_channels, double_z=False)
        self.backbone = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=z_channels, nhead=8, batch_first=True), num_layers=12, 
        )
        self.decoder = MLPDecoder(z_channels, 14)
        
        self.position_encoder = SinPositionEncoding(z_channels, max_sequence_length=65536)

        self.gs0_mean = nn.Linear(z_channels, z_channels)
        self.gs1_mean = nn.Linear(z_channels, z_channels)
        self.gs0_std = nn.Linear(z_channels, z_channels)
        self.gs1_std = nn.Linear(z_channels, z_channels)

    def reconstruct_loss(self, x, y):
        # compute target
        true_x, true_o, true_f, true_s, true_q = y.split([3, 1, 3, 3, 4], dim=-2)
        pred_x, pred_o, pred_f, pred_s, pred_q = x.split([3, 1, 3, 3, 4], dim=-2)

        loss_x = self.hparams.weight_x * (torch.norm(pred_x-true_x, dim=-2)).mean()

        loss_o = self.hparams.weight_o * torch.nn.functional.l1_loss(
            torch.sigmoid(pred_o),
            true_o
        )

        loss_f = self.hparams.weight_f * torch.nn.functional.l1_loss(pred_f, true_f)

        loss_s = self.hparams.weight_s * torch.nn.functional.l1_loss(torch.nn.functional.softplus(pred_s), true_s)

        loss_q = self.hparams.weight_q * (1-(torch.sum(torch.nn.functional.normalize(pred_q, dim=-1) * true_q, dim=-1).abs()).mean())

        loss =  loss_x + loss_o + loss_f + loss_s + loss_q

        return loss, (loss_x.item(), loss_o.item(), loss_f.item(), loss_s.item(), loss_q.item())

    def forward(self, target, indice):
        target = self.scale_gs.forward(target)
        target = self.sin_encoder(target.unsqueeze(-1)).flatten(-2, -1)
        target = self.encoder(target)

        B, s, _ = target.shape
        token = self.position_encoder(target)

        token = self.backbone(token, target)

        gs0_mean = self.gs0_mean(token)
        gs1_mean = self.gs1_mean(token)
        gs0_std = self.gs0_std(token)
        gs1_std = self.gs1_std(token)

        gs0_embed = gs0_mean + gs0_std * torch.randn_like(gs0_mean)
        gs1_embed = gs1_mean + gs1_std * torch.randn_like(gs1_mean)

        gs0 = self.decoder(gs0_embed)
        gs1 = self.decoder(gs1_embed)

        return torch.stack([gs0, gs1], dim=-1)
        
    @torch.no_grad()
    def inference(self, x):
        resi0, resi1, resi2, resi3, resi4, resi5 = self.forward(x)
        # shared token
        pos0 = torch.argmax(resi0, dim=-1, keepdim=True)
        pos1 = torch.argmax(resi1, dim=-1, keepdim=True)
        # token1
        pos2 = torch.argmax(resi2, dim=-1, keepdim=True)
        pos3 = torch.argmax(resi3, dim=-1, keepdim=True)
        # token2
        pos4 = torch.argmax(resi4, dim=-1, keepdim=True)
        pos5 = torch.argmax(resi5, dim=-1, keepdim=True)

        indices = [
            pos0, 
            pos1, 
            torch.concat([pos2, pos3], dim=-1), 
            torch.concat([pos4, pos5], dim=-1)
        ]
        return indices

    def process(self, x):
        # packed_pred = self.inference(x)

        gaussian = self.svqvae.idxBl_to_gaussian(x)

        flip_gaussian = torch.flip(gaussian, (0,))
        gaussian = gaussian[0]
        data = np.load("datasets/merge/0.npz")

        ardecoder = ARDecoder()
        for i, gs in enumerate(tqdm(gaussian)):
            ardecoder.add(gs)
            if i==1024:break
        
        xyz, opacity, features, scales, quats = ardecoder.get()
        xyz, opacity, features, scales, quats = activated_gs2gs(xyz, opacity, features, scales, quats)
        save_ply(f'decode-{i}.ply', xyz, opacity, features, scales, quats)

    def training_step(self, batch, batch_idx):
        source = batch[0]
        target = batch[1]
        indice = batch[2]

        pred_s = self.forward(target, indice)

        loss, loss_disp = self.reconstruct_loss(pred_s, source)

        self.log_dict({
            "recon_source/loss": loss,
            "recon_source/x": loss_disp[0],
            "recon_source/o": loss_disp[1],
            "recon_source/f": loss_disp[2],
            "recon_source/s": loss_disp[3],
            "recon_source/q": loss_disp[4],
        })

        return loss

    def configure_optimizers(self):
        # 示例: 配置优化器
        return torch.optim.Adam(self.parameters(), lr=0.0001)
