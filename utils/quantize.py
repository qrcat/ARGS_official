from .gaussian import norm_quats
from math import log2

import torch
import numpy as np

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
    def __init__(self, x_dims=8, o_dims=8, f_dims=8, s_dims=8, q_dims=8, log_left=False):
        self.C0 = 0.28209479177387814

        self.max_scale = 1.0
        self.min_scale = 2**-14

        self.x_grid_nums = 2**x_dims
        self.o_grid_nums = 2**o_dims
        self.f_grid_nums = 2**f_dims
        self.s_grid_nums = 2**s_dims
        self.q_grid_nums = 2**q_dims

        self.log_left = log_left

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
        return _round((features_dc * self.C0 + 0.5) * (self.f_grid_nums-1))
    
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
        quats_clamped = quats.clip(-1, 1)
        return _round((quats_clamped + 1) / 2 * (self.q_grid_nums-1))
    
    def _dequantize_q(self, q_indices):
        return (q_indices / (self.q_grid_nums-1) * 2 - 1)

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
    
if __name__ == '__main__':
    quantize = Quantize()
    # test for x
    x = np.linspace(-0.5, 0.5, 10000)
    x_i = quantize._quantize_x(x)
    d_x = quantize._dequantize_x(x_i)
    # test for o
    o = np.linspace(0, 1, 10000)
    o_i = quantize._quantize_o(o)
    d_o = quantize._dequantize_o(o_i)
    # test for f
    f = np.linspace(-1.772, 1.772, 10000)
    f_i = quantize._quantize_f(f)
    d_f = quantize._dequantize_f(f_i)
    # test fot s
    s = np.linspace(2**-16, 1, 10000)
    s_i = quantize._quantize_s(s)
    u_s_i, count_s_i = np.unique(s_i, return_counts=True)
    d_s = quantize._dequantize_s(s_i)
    u_s, count_s = np.unique(d_s, return_counts=True)
    # test for q
    q = np.linspace(-1, 1, 10000)
    q_i = quantize._quantize_q(q)
    d_q = quantize._dequantize_q(q_i)


    
