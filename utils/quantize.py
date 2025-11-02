from .gaussian import norm_quats
from math import log2

import torch
import numpy as np
try:
    import faiss
except:
    ...

def _round(x):
    """warper for round"""
    if isinstance(x, np.ndarray):
        return np.round(x).astype(np.int64)
    elif isinstance(x, torch.Tensor):
        return torch.round(x).long()
    else:
        return int(round(x))

def _exp2(x):
    """warper for exp2"""
    if isinstance(x, np.ndarray):
        return np.exp2(x)
    elif isinstance(x, torch.Tensor):
        return torch.exp2(x)
    else:
        return 2**x
    
def _log2(x):
    """warper for log2"""
    if isinstance(x, np.ndarray):
        return np.log2(x)
    elif isinstance(x, torch.Tensor):
        return torch.log2(x)
    else:
        return log2(x)
    
def _split(x, indices_or_sections, axis=-1):
    """warper for split"""
    if isinstance(x, np.ndarray):
        return np.split(x, indices_or_sections, axis=axis)
    elif isinstance(x, torch.Tensor):
        return torch.split(x, indices_or_sections, dim=axis)
    else:
        return x.split(indices_or_sections, axis=axis)

def _concat(*args):
    if isinstance(args[0], np.ndarray):
        return np.concatenate(args, axis=-1)
    elif isinstance(args[0], torch.Tensor):
        return torch.cat(args, dim=-1)
    else:
        return args[0].concat(args[1:])

class Quantize:
    def __init__(self, x_dims=8, o_dims=8, f_dims=8, s_dims=8, q_dims=8, log_left=False, value_base_q=True, dual_q_data=False):
        self.log_left = log_left
        self.value_base_q = value_base_q
        self.dual_q_data = dual_q_data

        self.C0 = 0.28209479177387814

        self.max_scale = 1.0
        self.min_scale = 2**-14

        self.x_grid_nums = 2**x_dims
        self.o_grid_nums = 2**o_dims
        self.f_grid_nums = 2**f_dims
        self.s_grid_nums = 2**s_dims
        if self.value_base_q:
            self.q_grid_nums = 2**q_dims
        else:
            self.ref_quats = np.load('utils/quaternions_quantize.npy')
            
            self.q_grid_nums = self.ref_quats.shape[0]

            self.ref_quats = self.ref_quats / np.linalg.norm(self.ref_quats, axis=1, keepdims=True)

            self.N, self.D = self.ref_quats.shape

            if self.dual_q_data:
                # 将 q 和 -q 都加入索引，以处理 ±q 等价
                db = np.concatenate([self.ref_quats, -self.ref_quats], axis=0).astype('float32')
            else:
                db = self.ref_quats.astype('float32')

            self.index = faiss.IndexFlatIP(self.D)  # 基于内积的索引
            self.index.add(db)

    def _quantize_x(self, x):
        x_clamped = x.clip(-0.5, 0.5)
        return _round((x_clamped + 0.5) * (self.x_grid_nums-1))
    
    def _dequantize_x(self, x_indices):
        return -0.5 + x_indices / (self.x_grid_nums-1)

    def _quantize_o(self, opacities):
        return _round(opacities * (self.o_grid_nums-1))
    
    def _dequantize_o(self, o_indices):
        return o_indices / (self.o_grid_nums-1)

    def _quantize_f(self, features_dc):
        return _round((features_dc * self.C0 + 0.5).clip(0, 1) * (self.f_grid_nums-1))
    
    def _dequantize_f(self, f_indices):
        return (f_indices / (self.f_grid_nums-1) - 0.5)/ self.C0
    
    def _quantize_s(self, scales):
        scale_clamped = scales.clip(self.min_scale, self.max_scale)
        if self.log_left:
            return _round((_log2(scale_clamped)-_log2(self.min_scale)) / (-_log2(self.min_scale)) * (self.s_grid_nums-1))
        else:
            return _round((_log2(scale_clamped+1)/_log2(self.max_scale+1)) * (self.s_grid_nums-1))
    
    def _dequantize_s(self, s_indices):
        if self.log_left:
            s = _exp2(s_indices / (self.s_grid_nums-1) * (-_log2(self.min_scale)) + _log2(self.min_scale))
        else:
            s = _exp2(s_indices / (self.s_grid_nums-1) * _log2(self.max_scale+1))-1

        return s.clip(min=self.min_scale, max=self.max_scale)

    def _quantize_q(self, quats):
        quats = norm_quats(quats)
        if self.value_base_q:
            quats_clamped = quats.clip(-1, 1)
            return _round((quats_clamped + 1) / 2 * (self.q_grid_nums-1))
        else:
            shape = quats.shape
            quats = quats.reshape(-1, 4)
            if isinstance(quats, np.ndarray):
                quats = quats.astype('float32')
            sims, idx = self.index.search(quats, 1)
            indices = _concat(idx//256, idx % 256)
            if isinstance(quats, np.ndarray):
                return indices.reshape(shape[:-1], 4)
            else:
                indices = _concat(torch.from_numpy(idx//256), torch.from_numpy(idx%256))
                return torch.from_numpy(indices).reshape(shape[:-1], 4)
    
    def _dequantize_q(self, q_indices):
        if self.value_base_q:
            return (q_indices / (self.q_grid_nums-1) * 2 - 1)
        else:
            q_indices = q_indices[..., 0] * 256 + q_indices[..., 1]
            return self.ref_quats[q_indices]

    def quantize(self, x, o, f, s, q):
        x_indices = self._quantize_x(x)
        o_indices = self._quantize_o(o)
        f_indices = self._quantize_f(f)
        s_indices = self._quantize_s(s)
        q_indices = self._quantize_q(q)
        return x_indices, o_indices, f_indices, s_indices, q_indices
    
    def dequantize(self, x_indices, o_indices, f_indices, s_indices, q_indices):
        x = self._dequantize_x(x_indices)
        o = self._dequantize_o(o_indices)
        f = self._dequantize_f(f_indices)
        s = self._dequantize_s(s_indices)
        q = self._dequantize_q(q_indices)
        return x, o, f, s, q
    
    @torch.no_grad()
    def __call__(self, x, o=None, f=None, s=None, q=None):
        if x is not None and o is None and f is None and s is None and q is None:
            x, o, f, s, q = _split(x, [3, 1, 3, 3, 4], axis=-1)
            merge = True
        else:
            merge = False
        x_indices, o_indices, f_indices, s_indices, q_indices = self.quantize(x, o, f, s, q)
        x, o, f, s, q = self.dequantize(x_indices, o_indices, f_indices, s_indices, q_indices)
        
        if merge:
            x = _concat(x, o, f, s, q)
            return x
        else:
            return x, o, f, s, q

    @torch.no_grad()
    def get_indices(self, x, o=None, f=None, s=None, q=None):
        if x is not None and o is None and f is None and s is None and q is None:
            x, o, f, s, q = _split(x, [3, 1, 3, 3, 4], axis=-1)
            merge = True
        else:
            merge = False
        x_indices, o_indices, f_indices, s_indices, q_indices = self.quantize(x, o, f, s, q)
        
        if merge:
            x = _concat(x_indices, o_indices, f_indices, s_indices, q_indices)
            return x
        else:
            return x_indices, o_indices, f_indices, s_indices, q_indices
    
def fibonacci_quaternion_sampling(n_points: int):
    # (Saff & Kuijlaars 1997)
    phi = (3.0 - np.sqrt(5.0)) * np.pi  # golden angle
    psi = (3.0 - np.sqrt(7.0)) * np.pi

    quats = np.zeros((n_points, 4))
    for i in range(n_points):
        u = i / (n_points - 1)
        eta = np.arcsin(np.sqrt(u))
        phi1 = phi * i
        phi2 = psi * i
        
        w = np.cos(eta) * np.cos(phi1)
        x = np.cos(eta) * np.sin(phi1)
        y = np.sin(eta) * np.cos(phi2)
        z = np.sin(eta) * np.sin(phi2)

        quats[i] = [w, x, y, z]

    return quats / np.linalg.norm(quats, axis=1, keepdims=True)
