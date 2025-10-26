from utils.render import camera_matrix_from_angles, load_ply_torch, tensor_from_numpy
from utils.shs import SH2RGB
from utils.io import  gs2activated_gs, load_ply, activated_gs2gs
from torchvision.transforms.functional import normalize
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import argparse
import gsplat
import pickle
import torch
import numpy as np
import time


def get_camera_params(radius, device='cuda'):
    width, height, focal = 224, 224, 200.0
    Ks = torch.tensor([[focal, 0.0, width/2.0], [0.0, focal, height/2.0], [0.0, 0.0, 1.0],], device=device)[None]

    camera_pair_list = [
        (0, 30), (45, 30), (90, 30), (135, 30), (180, 30), (225, 30), (270, 30), (315, 30)
    ]

    cam_pos = []
    for azi, ele in camera_pair_list:
        view_mat = camera_matrix_from_angles(azi/180*np.pi, ele/180*np.pi, radius, np.array([0.0, 0.0, -1.0]))
        cam_pos.append(view_mat)
    viewmats = np.stack(cam_pos)
    viewmats = torch.from_numpy(viewmats).to(device, dtype=torch.float)
    
    B, _, _ = viewmats.shape

    return viewmats, Ks.repeat(B, 1, 1), width, height


def compare(input_ply: str, input_pkl: str, device='cuda'):
    xyz, opacities, features_dc, scales, rots = load_ply_torch(input_ply, device)

    features_dc = SH2RGB(features_dc)

    radius   = xyz.norm(dim=1).max().item()*2

    viewmats, Ks, width, height = get_camera_params(radius, device)

    # render image
    rets = gsplat.rasterization(xyz, rots, scales, opacities[:, 0], features_dc, viewmats, Ks, width, height)
    gt   = torch.cat(rets[:2], dim=-1)

    with open(input_pkl, 'rb') as f:
        databack = pickle.load(f)

    data = {}

    for level in range(databack['level']):
        rets = databack['data'][databack[level][0]]
        rets = torch.from_numpy(rets).to(device, dtype=torch.float)

        xyz, opacities, features_dc, scales, rots =  rets.split([3, 1, 3, 3, 4], dim=-1)

        features_dc = SH2RGB(features_dc)

        rets = gsplat.rasterization(xyz, rots, scales, opacities[:, 0], features_dc, viewmats, Ks, width, height)
        pred = torch.cat(rets[:2], dim=-1)

        mse  = torch.nn.functional.mse_loss(pred, gt)
        psnr = 20 * np.log10(1 / mse.sqrt().item())

        data[level] = psnr

    print(data)


        

if __name__ == '__main__':
    compare('point_cloud.ply', 'data.pkl')