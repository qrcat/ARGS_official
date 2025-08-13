from .general import log, exp, sigmoid, inv_sigmoid, softplus, inv_softplus
from .gaussian import build_sigma, norm_quats
import torch
import numpy as np
from plyfile import PlyData, PlyElement


"""
The following code is used to convert between different representations of the Gaussian parameters.
"""
def pack_gaussian_parameters(*gaussian_parameters):
    if isinstance(gaussian_parameters[0], torch.Tensor):
        return torch.concat(gaussian_parameters, dim=-1)
    elif isinstance(gaussian_parameters[0], np.ndarray):
        return np.concatenate(gaussian_parameters, axis=-1)
    else:
        raise ValueError(f"gaussian_parameters[0] = {gaussian_parameters[0]}")
    
def unpack_gaussian_parameters(*gaussian_parameters):
    if len(gaussian_parameters) == 1:
        gaussian_parameters = gaussian_parameters[0]
        assert gaussian_parameters.shape[-1] == 14

        xyz = gaussian_parameters[..., :3]
        opacities = gaussian_parameters[..., 3:4]
        features_dc = gaussian_parameters[..., 4:7]
        scales = gaussian_parameters[..., 7:10]
        rots = gaussian_parameters[..., 10:14]
    elif len(gaussian_parameters) == 5:
        xyz, opacities, features_dc, scales, rots = gaussian_parameters
    else:
        raise ValueError(f"len(gaussian_parameters) = {len(gaussian_parameters)}")
    
    return xyz, opacities, features_dc, scales, rots

def get_combinable_gaussian(xyz, opacities, features_dc, scales, rots):
    xyz = xyz
    opacities = sigmoid(opacities)
    features_dc = features_dc
    scales = exp(scales)
    rots = norm_quats(rots)

    sigma, inv_sigma = build_sigma(scales, rots)

    return xyz, opacities, features_dc, sigma, inv_sigma

def gs2activated_gs(*gaussian_parameters):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(*gaussian_parameters)

    opacities = sigmoid(opacities)
    scales = exp(scales)

    return xyz, opacities, features_dc, scales, rots

def train_gs2activated_gs(*gaussian_parameters, beta=1.0):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(*gaussian_parameters)
    
    opacities = sigmoid(opacities)
    scales = softplus(scales, beta=beta)

    if len(gaussian_parameters) == 1:
        return pack_gaussian_parameters(xyz, opacities, features_dc, scales, rots)
    elif len(gaussian_parameters) == 5:
        return xyz, opacities, features_dc, scales, rots

def activated_gs2gs(*gaussian_parameters, delta=1e-6):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(*gaussian_parameters)

    opacities = inv_sigmoid(opacities.clip(min=delta, max=1.0-delta))
    scales = log(scales.clip(min=delta))

    if len(gaussian_parameters) == 1:
        return pack_gaussian_parameters(xyz, opacities, features_dc, scales, rots)
    elif len(gaussian_parameters) == 5:
        return xyz, opacities, features_dc, scales, rots

def activated_gs2train_gs(*gaussian_parameters, delta=1e-10, beta=1.0):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(*gaussian_parameters)
    
    opacities = inv_sigmoid(opacities.clip(min=delta, max=1.0-delta))
    scales = inv_softplus(scales.clip(min=delta), beta=beta)

    if len(gaussian_parameters) == 1:
        return pack_gaussian_parameters(xyz, opacities, features_dc, scales, rots)
    elif len(gaussian_parameters) == 5:
        return xyz, opacities, features_dc, scales, rots

"""
The following code is used to exchange with files.
"""
def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def convert2numpy(*args):
    return [a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in args]

def convert2torch(*args, device=None):
    return [torch.from_numpy(a).to(device) if isinstance(a, np.ndarray) else a for a in args]

def save_ply(path, xyz, opacities, features_dc, scales, rots):
    xyz, opacities, features_dc, scales, rots = convert2numpy(xyz, opacities, features_dc, scales, rots)
    normals = np.zeros_like(xyz)

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    attributes = np.concatenate((xyz, normals, features_dc, opacities, scales, rots), axis=1).astype(np.float32)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, opacities, features_dc, scales, rots
