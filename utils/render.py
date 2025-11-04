from .io import  gs2activated_gs, load_ply
import numpy as np
import torch

def tensor_from_numpy(np_ls: list, device):
    return [torch.from_numpy(np_arr).to(device, dtype=torch.float) for np_arr in np_ls]

def load_ply_torch(path: str, device):
    rets = gs2activated_gs(*load_ply(path))
    return tensor_from_numpy(rets, device)

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))

    view_mat = np.eye(4)
    view_mat[:3, 0] = vec0
    view_mat[:3, 1] = vec1
    view_mat[:3, 2] = vec2
    view_mat[:3, 3] = position

    return view_mat

def camera_matrix_from_angles(azimuth: float, elevation: float, radius: float = 1.0, up: np.ndarray = None, axis = 'z') -> np.ndarray:
    """
    根据方位角和俯仰角计算相机的视图矩阵（从相机看向原点）。
    
    参数:
        azimuth (float): 方位角（弧度），从 x 轴开始在 xy 平面逆时针旋转
        elevation (float): 俯仰角（弧度），从 xy 平面起算，向上为正
        radius (float): 相机到原点的距离
        up (np.ndarray): 向上方向，默认为 (0, 0, 1)
    
    返回:
        np.ndarray: 4x4 视图矩阵
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    
    # 1. 从角度计算相机位置
    cos_e = np.cos(elevation)
    sin_e = np.sin(elevation)
    cos_a = np.cos(azimuth)
    sin_a = np.sin(azimuth)
    
    # 相机位置（在球坐标系中）
    if axis == 'x':
        ...
    elif axis == 'y':
        position = np.array([
            radius * cos_e * cos_a,
            radius * sin_e,
            radius * cos_e * sin_a,
        ])
    else:
        position = np.array([
            radius * cos_e * cos_a,
            radius * cos_e * sin_a,
            radius * sin_e
        ])
    
    # 2. 观察方向：从相机指向原点
    lookdir = -position  # 指向原点
    
    # 3. 使用 viewmatrix 构建视图矩阵
    view_mat = np.linalg.inv(viewmatrix(lookdir, up, position))
    
    return view_mat
