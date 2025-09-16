from utils.render import camera_matrix_from_angles
from utils.io import activated_gs2train_gs
from pgs import PGSMoments
from utils.shs import SH2RGB
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
import os
import cv2
from pathlib import Path
from tqdm import tqdm
from utils.io import get_combinable_gaussian, activated_gs2gs, activated_gs2train_gs, gs2activated_gs, save_ply, load_ply
import torch
import gsplat
from torchvision.utils import save_image
from pathlib import Path
import numpy as np


def load_ply_torch(path: str):
    xyz, opacities, features_dc, scales, rots = gs2activated_gs(*load_ply(path))
    xyz = torch.as_tensor(xyz, dtype=torch.float, device='cuda')
    opacities = torch.as_tensor(opacities, dtype=torch.float, device='cuda')
    features_dc = torch.as_tensor(features_dc, dtype=torch.float, device='cuda')
    scales = torch.as_tensor(scales, dtype=torch.float, device='cuda')
    rots = torch.as_tensor(rots, dtype=torch.float, device='cuda')
    return xyz, opacities, features_dc, scales, rots


items = [
    "airplane", "bed", "bookshelf", "bowl", "chair", "cup", "desk", "dresser", "glass_box", "keyboard", "laptop", "monitor", "person", "plant", "range_hood", "sofa", "stool", "tent", "tv_stand", "wardrobe",
    "bathtub", "bench", "bottle", "car", "cone", "curtain", "door", "flower_pot", "guitar", "lamp", "mantel", "night_stand", "piano", "radio", "sink", "stairs", "table", "toilet", "vase", "xbox"
]

parser = argparse.ArgumentParser()
parser.add_argument("--modelsplat_ply", type=str, required=True, default="/mnt/private_rqy/gs_data/modelsplat_ply")
args = parser.parse_args()

# shutil.rmtree("output")
os.mkdir("output")
for item in tqdm(items):
    os.mkdir(f"output/{item}")

    pgs = PGSMoments.load(os.path.join(args.modelsplat_ply, f"/{item}/train/{item}_0001/point_cloud.ply"))
    for size in [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        if os.path.exists(f"output/{item}/zip-{size:06d}.ply"):
            continue
        pgs.simplify(size)
        pgs.save(f"output/{item}/zip-{pgs.used_size:06d}.ply")

    path = Path(f'output/{item}')
    plys = sorted(path.glob('*.ply'))

    width, height = 1920, 1080
    Ks = torch.tensor(
        [
            [1000.0, 0.0, width/2.0],
            [0.0, 1000.0, height/2.0],
            [0.0, 0.0, 1.0],
        ], device='cuda'
    )[None]

    azi = 0
    ele = 30
    for ply in plys:
        xyz, opacities, features_dc, scales, rots = load_ply_torch(ply)
        features_dc = SH2RGB(features_dc)
        
        gs_size = int(ply.stem.split('-')[-1])

        azi = azi % 360
        ele = ele % 360
        for i in range(6+int(np.log2(gs_size))*2):
            azi = azi + 10
            view_mat = np.linalg.inv(camera_matrix_from_angles(azi/180*np.pi, theta/180*np.pi, 1.4, np.array([0.0, 0.0, -1.0])))
            viewmats = torch.tensor(view_mat.tolist(), device='cuda')[None]

            output = gsplat.rasterization(xyz, rots, scales, opacities[:, 0], features_dc, viewmats, Ks, width, height)
        
            image = output[0][0]
            alpha = output[1][0]

            rgba = torch.cat([image, alpha], dim=-1)
            save_image(rgba.permute(2, 0, 1), ply.with_suffix(f'.{azi:03d}.{theta:03d}.png'))

    images = sorted(path.glob("*.png"))

    # 读取第一张图片，获取视频帧的尺寸
    first_image_path = images[0]
    frame = cv2.imread(first_image_path.as_posix())
    if frame is None:
        raise ValueError(f"无法读取第一张图片: {first_image_path}")
    height, width, layers = frame.shape
    size = (width, height)

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter((path/f'{item}.m4v').as_posix(), fourcc, fps, size)

    for fname in tqdm(images):
        fname = fname.as_posix()
        metas = fname.split('/')[-1].split('-')[-1]
        gs_size, azi, theta = metas.split('.')[:-1]
        frame = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if frame is not None:
            image, alpha = frame[:, :, :3], frame[:, :, 3:]
            # fill background with white color
            image = image + (255-alpha)
            image = cv2.putText(image, f'nums={gs_size}', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (56, 56, 56), 3, cv2.LINE_AA)
            image = cv2.putText(image, f'azi={azi}', (100, 180), cv2.FONT_HERSHEY_COMPLEX, 2, (56, 56, 56), 3, cv2.LINE_AA)
            image = cv2.putText(image, f'theta={theta}', (100, 260), cv2.FONT_HERSHEY_COMPLEX, 2, (56, 56, 56), 3, cv2.LINE_AA)

            video_writer.write(image)  # 写入帧
        else:
            print(f"警告: 无法读取图片 {fname}, 已跳过.")

    video_writer.release()
