from utils.io import activated_gs2train_gs
from pgs import PGSMoments
from utils.shs import SH2RGB
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import argparse
import time
import shutil
import os
import cv2
from utils.render import camera_matrix_from_angles_y
from pathlib import Path
from tqdm import tqdm
from utils.io import get_combinable_gaussian, activated_gs2gs, activated_gs2train_gs, gs2activated_gs, save_ply, load_ply

import torch
import gsplat
from torchvision.utils import save_image
from pathlib import Path


def load_ply_torch(path: str):
    xyz, opacities, features_dc, scales, rots = gs2activated_gs(*load_ply(path))

    return xyz, opacities, features_dc, scales, rots


item = 'bird'
path = Path(f'output/{item}')
path.mkdir(exist_ok=True)

# render config
width, height = 1920, 1080
# camera parameters
Ks = torch.tensor(
    [
        [1000.0, 0.0, width/2.0],
        [0.0, 1000.0, height/2.0],
        [0.0, 0.0, 1.0],
    ], device='cuda'
)[None]
# init cameras

azi = 0
ele = 30

_level = 14
_sizes = 2**np.arange(0, _level+1)
_thres = np.cumsum(_sizes*32)
# load data
pgs = PGSMoments.load("gradio_output.ply")
# get init scales
for i in range(_level):
    if pgs.used_size < _thres[i]:
        next_size = pgs.used_size - _sizes[0] if i == 0 else (pgs.used_size - _thres[i-1]) // _sizes[i] * _sizes[i] + _thres[i-1]
        pgs.simplify(next_size)
        break
# simplify
while pgs.used_size > 1:
    for i in range(_level):
        if pgs.used_size < _thres[i]:
            pgs.simplify(pgs.used_size - _sizes[i])
            break
    # breakpoint()
    xyz, opacities, features_dc, scales, rots = pgs.get()
    # convert to tensor
    xyz = torch.as_tensor(xyz, dtype=torch.float, device='cuda')
    opacities = torch.as_tensor(opacities, dtype=torch.float, device='cuda')
    features_dc = torch.as_tensor(features_dc, dtype=torch.float, device='cuda')
    scales = torch.as_tensor(scales, dtype=torch.float, device='cuda')
    rots = torch.as_tensor(rots, dtype=torch.float, device='cuda')
    # convert to rgb
    features_dc = SH2RGB(features_dc)
    azi = (azi+10) % 360
    # get camera matrix
    view_mat = np.linalg.inv(camera_matrix_from_angles_y(azi/180*np.pi, ele/180*np.pi, 1.4, np.array([0.0, -1.0, 0.0])))
    viewmats = torch.tensor(view_mat.tolist(), device='cuda')[None]
    # render
    output = gsplat.rasterization(xyz, rots, scales, opacities[:, 0], features_dc, viewmats, Ks, width, height)
    
    image = output[0][0]
    alpha = output[1][0]

    rgba = torch.cat([image, alpha], dim=-1)
    save_image(rgba.permute(2, 0, 1), path / f"{pgs.used_size:06d}.png")

# render video
images = sorted(path.glob("*.png"))

# 读取第一张图片，获取视频帧的尺寸
first_image_path = images[0]
frame = cv2.imread(first_image_path.as_posix())
if frame is None:
    raise ValueError(f"无法读取第一张图片: {first_image_path}")
height, width, layers = frame.shape
size = (width, height)

fps = 60
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter((path/'bird.m4v').as_posix(), fourcc, fps, size)

for fname in tqdm(images):
    fname = fname.as_posix()
    metas = fname.split('/')[-1].split('-')[-1]
    gs_size, = metas.split('.')[:-1]
    frame = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if frame is not None:
        image, alpha = frame[:, :, :3], frame[:, :, 3:]
        # fill background with white color
        image = image + (255-alpha)
        image = cv2.putText(image, f'nums={gs_size}', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (56, 56, 56), 3, cv2.LINE_AA)

        video_writer.write(image)  # 写入帧
    else:
        print(f"警告: 无法读取图片 {fname}, 已跳过.")

video_writer.release()
