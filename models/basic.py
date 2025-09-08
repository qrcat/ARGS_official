import lightning as L
import torch.nn.functional as F
import torch.nn as nn
import torch


class ScaleGaussianSplat(nn.Module):
    """Limit the input into [-1, 1]"""
    def __init__(
        self, 
        avg_x=0.0, std_x=1.0, 
        avg_o=0.0, std_o=1.0,
        avg_f=0.0, std_f=0.5/0.28209479177387814,
        avg_s=0.0, std_s=1.0,
        avg_q=0.0, std_q=1.0,
    ):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(
            [
                avg_x, avg_x, avg_x, 
                avg_o, 
                avg_f, avg_f, avg_f, 
                avg_s, avg_s, avg_s,
                avg_q, avg_q, avg_q, avg_q,
            ]
        ))

        self.register_buffer('std', torch.as_tensor(
            [
                std_x, std_x, std_x, 
                std_o, 
                std_f, std_f, std_f, 
                std_s, std_s, std_s,
                std_q, std_q, std_q, std_q,
            ]
        ))

    def forward(self, gaussian):
        gaussian = (gaussian-self.mean)/self.std
        return gaussian
    
    def inverse(self, gaussian):
        gaussian = gaussian*self.std+self.mean
        return gaussian


class BasicVAE(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.pos_act = lambda x: 0.5 * torch.tanh(x)
        self.scale_act = self.activate_scale
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x, dim: F.normalize(x, dim=dim)
    
    @staticmethod
    def activate_scale(scale_before_activated, indice):
        scale_mul = BasicVAE.scale_mul(indice)
        return scale_mul[..., None, None] * 0.1 * F.softplus(scale_before_activated)
    
    def to_activated(self, x, indices):
        x[..., :3] = self.pos_act(x[..., :3])
        x[..., 3:4] = self.opacity_act(x[..., 3:4])
        pass # 4:7
        x[..., 7:10] = self.activate_scale(x[..., 7:10], indices)
        x[..., 10:] = self.rot_act(x[..., 10:], dim=-1)
        return x
    
    @staticmethod
    def scale_mul(indice):
        return 0.9 * (1-torch.log2(indice+1).clip(max=16)/16) + 0.1

    # =============================== only used on training ===============================
    def reconstruct_loss(self, x, y, indice, mask=None, gs_dims=-1, weights=None):
        true_x, true_o, true_f, true_s, true_q = y.split([3, 1, 3, 3, 4], dim=gs_dims)
        pred_x, pred_o, pred_f, pred_s, pred_q = x.split([3, 1, 3, 3, 4], dim=gs_dims)
        # here, we use the scale_mul for the true scale in training
        # scale_mul = self.scale_mul(indice)
        # true_s = true_s/scale_mul[..., None, None]

        pred_x = self.pos_act(pred_x)
        pred_o = self.opacity_act(pred_o)
        # here, we don't use the scale_mul for the predicted scale
        # but in the eval, we need to multiply it back
        pred_s = 0.1 * F.softplus(pred_s)
        pred_q = self.rot_act(pred_q, dim=gs_dims)

        if mask is not None:
            pred_x = pred_x[mask]
            pred_o = pred_o[mask]
            pred_f = pred_f[mask]
            pred_s = pred_s[mask]
            pred_q = pred_q[mask]

            true_x = true_x[mask]
            true_o = true_o[mask]
            true_f = true_f[mask]
            true_s = true_s[mask]
            true_q = true_q[mask]

        loss_x = torch.nn.functional.l1_loss(pred_x, true_x)
        loss_o = torch.nn.functional.l1_loss(pred_o, true_o)
        loss_f = torch.nn.functional.l1_loss(pred_f, true_f)
        loss_s = torch.nn.functional.l1_loss(pred_s, true_s)

        dot_product = torch.sum(pred_q * true_q, dim=gs_dims).abs()
        loss_q = 1 - dot_product.mean()

        if weights is not None:
            loss = weights[0] * loss_x + weights[1] * loss_o + weights[2] * loss_f + weights[3] * loss_s + weights[4] * loss_q
        else:
            loss = loss_x + loss_o + loss_f + loss_s + loss_q
        
        return loss, (loss_x.item(), loss_o.item(), loss_f.item(), loss_s.item(), loss_q.item())
    # =====================================================================================

    def compute_SNR(self, x, y, peak=None):
        signal = y**2 if peak is None else peak
        noises = (x - y)**2
        return 20 * torch.log10((signal / noises).clip(1e-6, 1e6)).mean().item()

    @torch.no_grad()
    def eval_reconstruct(self, x, y, indice, mask=None, gs_dims=-1):
        # compute target
        true_x, true_o, true_f, true_s, true_q = y.split([3, 1, 3, 3, 4], dim=gs_dims)
        pred_x, pred_o, pred_f, pred_s, pred_q = x.split([3, 1, 3, 3, 4], dim=gs_dims)

        pred_x = self.pos_act(pred_x)
        pred_o = self.opacity_act(pred_o)
        # pred_s = self.scale_act(pred_s, indice)
        pred_s = 0.1 * F.softplus(pred_s)
        pred_q = self.rot_act(pred_q, dim=gs_dims)

        if mask is not None:
            pred_x = pred_x[mask]
            pred_o = pred_o[mask]
            pred_f = pred_f[mask]
            pred_s = pred_s[mask]
            pred_q = pred_q[mask]

            true_x = true_x[mask]
            true_o = true_o[mask]
            true_f = true_f[mask]
            true_s = true_s[mask]
            true_q = true_q[mask]

        metrics = {}

        metrics['xyz_l1'] = torch.nn.functional.l1_loss(pred_x, true_x).item()
        metrics['xyz_SNR'] = self.compute_SNR(pred_x, true_x, peak=1.0)
        metrics['xyz_sphere_distance'] = (torch.norm(pred_x-true_x, dim=gs_dims)).mean().item()

        metrics['opacity_l1'] = torch.nn.functional.l1_loss(pred_o, true_o).item()
        metrics['opacity_SNR'] = self.compute_SNR(pred_o, true_o, peak=1.0)

        metrics['feature_l1'] = torch.nn.functional.l1_loss(pred_f, true_f).item()
        metrics['feature_SNR'] = self.compute_SNR(pred_f, true_f, peak=1.78)

        metrics['scale_l1'] = torch.nn.functional.l1_loss(pred_s, true_s).item()
        metrics['scale_SNR'] = self.compute_SNR(pred_s, true_s)

        metrics['quat_l1'] = torch.nn.functional.l1_loss(pred_q, true_q).item()
        metrics['quat_distance'] = 1-(torch.sum(pred_q * true_q, dim=gs_dims).abs()).mean().item()
        metrics['quat_SNR'] = self.compute_SNR(pred_q, true_q)

        return metrics
