from .quaternion import quaternion_to_matrix, matrix_to_quaternion, normalize_quaternions
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

def build_sigma(scale, quat):
    if isinstance(scale, torch.Tensor):
        rotate_matrix = quaternion_to_matrix(quat)
        scales_matrix = torch.zeros_like(rotate_matrix)

        scales_matrix[..., 0, 0] = scale[..., 0]
        scales_matrix[..., 1, 1] = scale[..., 1]
        scales_matrix[..., 2, 2] = scale[..., 2]

        sigma = (rotate_matrix @ scales_matrix @ scales_matrix.transpose(-2, -1) @ rotate_matrix.transpose(-2, -1))
        inv_sigma = torch.linalg.inv(sigma)
    elif isinstance(scale, np.ndarray):
        rotate_matrix = R.from_quat(quat).as_matrix()
        scales_matrix = np.zeros_like(rotate_matrix)

        scales_matrix[..., 0, 0] = scale[..., 0]
        scales_matrix[..., 1, 1] = scale[..., 1]
        scales_matrix[..., 2, 2] = scale[..., 2]

        sigma = (rotate_matrix @ scales_matrix @ scales_matrix.transpose(*np.arange(scales_matrix.ndim-2), -1, -2) @ rotate_matrix.transpose(*np.arange(rotate_matrix.ndim-2), -1, -2))
        inv_sigma = np.linalg.inv(sigma)

    return sigma, inv_sigma

def unpack_sigma(sigma):
    if isinstance(sigma, torch.Tensor):
        _Rm, _sp, _Rmt = torch.linalg.svd(sigma)
        _scale = torch.sqrt(_sp)
        if len(sigma) == 2:
            if torch.linalg.det(_Rm) < 0: _Rm = -_Rm # fix the sign
        else:
            mask = torch.linalg.det(_Rm) < 0
            _Rm = torch.where(mask, -_Rm, _Rm)
        _quat = matrix_to_quaternion(_Rm)
    elif isinstance(sigma, np.ndarray):
        _Rm, _sp, _Rmt = np.linalg.svd(sigma)
        _scale = np.sqrt(_sp)
        if len(sigma) == 2:
            if np.linalg.det(_Rm) < 0: _Rm = -_Rm # fix the sign
        else:
            mask = np.linalg.det(_Rm) < 0
            _Rm[mask] = -_Rm[mask]
        _quat = R.from_matrix(_Rm).as_quat()
    else:
        raise TypeError(f"sigma must be torch.Tensor or np.ndarray, but got {type(sigma)}")

    return _scale, _quat

def norm_quats(quats):
    if isinstance(quats, torch.Tensor):
        return normalize_quaternions(quats)
    elif isinstance(quats, np.ndarray):
        quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
        return np.where(quats[..., 0:1] < 0, -quats, quats)
