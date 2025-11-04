from utils.render import camera_matrix_from_angles
from utils.io import save_ply, activated_gs2gs
from utils.shs import SH2RGB
from torchvision.utils import save_image
import numpy as np
import torch
import gsplat
import pickle

with open('data.pkl','rb') as f:
    data = pickle.load(f)

device = 'cuda'

camera_pair_list = [
    # (0, 30), 
    (45, 30),
    #  (90, 30), (135, 30), (180, 30), (225, 30), (270, 30), (315, 30)
]
cam_pos = []
for azi, ele in camera_pair_list:
    radius   = 2.0
    view_mat = camera_matrix_from_angles(azi/180*np.pi, ele/180*np.pi, radius, np.array([0.0, -1.0, 0.0]), 'y')
    cam_pos.append(view_mat)
viewmats = np.stack(cam_pos)
viewmats = torch.from_numpy(viewmats).to(device, dtype=torch.float)

B, _, _ = viewmats.shape
for level in range(0, data['level']):
    gs = torch.from_numpy(data['data'][data[level][0]]).float().to(device)
    print(level, gs.shape[0])

    width, height, focal = 1024, 1024, 1600.0
    Ks = torch.tensor([[focal, 0.0, width/2.0], [0.0, focal, height/2.0], [0.0, 0.0, 1.0],], device=device)[None]

    xyz, opacities, features_dc, scales, rots = gs.split([3, 1, 3, 3, 4], dim=-1)
    features_dc = SH2RGB(features_dc)
    output = gsplat.rasterization(xyz, rots, scales, opacities[:, 0], features_dc, viewmats, Ks.repeat(B, 1, 1), width, height)
    image, alpha = output[:2]
    rgba = torch.cat([image, alpha], dim=-1)

    for i in range(B):
        azi, ele = camera_pair_list[i]
        output_img =  f'test_{level}_{azi}_{ele}.png'
        save_image(rgba[i].permute(2, 0, 1), output_img)
