import numpy as np

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

def sigmoid_np(x):
    return 1/(1+np.exp(-x))

def inv_sigmoid_np(x):
    return np.log(x/(1-x))

def softplus_np(x, beta=1.0, threshold=20.0):
    if isinstance(x, np.ndarray):
        y = np.zeros_like(x)
        mask = x < threshold
        y[mask] = np.log(1+np.exp(beta*x[mask]))/beta
        y[~mask] = x[~mask]
        return y
    else:
        if x < threshold:
            return np.log(1+np.exp(beta*x))/beta
        else:
            return x

def inv_softplus_np(x, beta=1.0, threshold=20.0):
    return np.log(np.exp(beta*x)-1)/beta

def get_sigma(scales, rots):
    rotation = R.from_quat(rots)
    rotate_matrix = rotation.as_matrix()
    scales_matrix = np.zeros_like(rotate_matrix)

    scales_matrix[:, 0, 0] = scales[:, 0]
    scales_matrix[:, 1, 1] = scales[:, 1]
    scales_matrix[:, 2, 2] = scales[:, 2]

    sigma = (rotate_matrix @ scales_matrix @ scales_matrix.transpose(0, 2, 1) @ rotate_matrix.transpose(0, 2, 1))
    inv_sigma = np.linalg.inv(sigma)

    return sigma, inv_sigma

def get_combinable_gaussian(xyz, opacities, features_dc, scales, rots):
    xyz = xyz
    opacities = sigmoid_np(opacities)
    features_dc = features_dc
    scales = np.exp(scales)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)

    sigma, inv_sigma = get_sigma(scales, rots)

    return xyz, opacities, features_dc, sigma, inv_sigma

class GSItem(object):
    def __init__(self, xyz, opacity, features, Sigma, invSigma, index):
        self.xyz = xyz.astype(np.float64)
        self.opacity = opacity.astype(np.float64)
        self.features = features.astype(np.float64)
        self.Sigma = Sigma.astype(np.float64)
        self.invSigma = invSigma.astype(np.float64)
        self.index = index

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, i):
        return self.xyz[i]

    def __repr__(self):
        return 'Item({})'.format(self.xyz)

    def __lt__(self, other):
        return np.linalg.det(self.Sigma) < np.linalg.det(other.Sigma)
    
    def json(self):
        Rm, sq, Rmt = np.linalg.svd(self.Sigma)
        if np.linalg.det(Rm)<0:
            Rm = -Rm
        rotation = R.from_matrix(Rm)
        quat = rotation.as_quat()
        scales = np.sqrt(sq)

        return {
            "xyz": self.xyz,
            "opacity": self.opacity,
            "features": self.features,
            "scales": scales,
            "quat": quat,
        }

    def vector(self):
        data = self.json()
        return np.concatenate([data['xyz'], data['opacity'], data['features'], data['scales'], data['quat']])

def unpack_gaussian_parameters(gaussian_parameters):
    if len(gaussian_parameters) == 1:
        gaussian_parameters = gaussian_parameters[0]
        assert gaussian_parameters.shape[-1] == 14

        xyz = gaussian_parameters[:, :3]
        opacities = gaussian_parameters[:, 3:4]
        features_dc = gaussian_parameters[:, 4:7]
        scales = gaussian_parameters[:, 7:10]
        rots = gaussian_parameters[:, 10:14]
    elif len(gaussian_parameters) == 5:
        xyz, opacities, features_dc, scales, rots = gaussian_parameters
    else:
        raise ValueError(f"len(gaussian_parameters) = {len(gaussian_parameters)}")
    
    return xyz, opacities, features_dc, scales, rots

def train_gs_to_activated_gs(*gaussian_parameters, beta=1.0):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(gaussian_parameters)
    
    opacities = sigmoid_np(opacities)
    scales = softplus_np(scales, beta=beta)

    return xyz, opacities, features_dc, scales, rots

def gs_to_activated_gs(*gaussian_parameters):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(gaussian_parameters)

    opacities = sigmoid_np(opacities)
    scales = np.exp(scales)

    return xyz, opacities, features_dc, scales, rots

def activated_gs_to_train_gs(*gaussian_parameters, delta=1e-10, beta=1.0):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(gaussian_parameters)
    
    opacities = inv_sigmoid_np(opacities.clip(min=delta, max=1.0-delta))
    scales = inv_softplus_np(scales.clip(min=delta), beta=beta)

    return xyz, opacities, features_dc, scales, rots

def activated_gs_to_gs(*gaussian_parameters, delta=1e-10):
    xyz, opacities, features_dc, scales, rots = unpack_gaussian_parameters(gaussian_parameters)

    opacities = inv_sigmoid_np(opacities.clip(min=delta, max=1.0-delta))
    scales = np.log(scales.clip(min=delta))

    return xyz, opacities, features_dc, scales, rots
