from utils.render import camera_matrix_from_angles
from utils.io import activated_gs2train_gs
from pgs import PGSMoments
from utils.shs import SH2RGB
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from utils.io import gs2activated_gs, load_ply

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


parser = argparse.ArgumentParser()
parser.add_argument("--shapesplat_ply", type=str, required=True, default="/mnt/private_rqy/gs_data/shapesplat_ply/")
args = parser.parse_args()

items = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03046257', '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134', '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244', '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04401088', '04460130', '04468005', '04530566', '04554684']

for item in tqdm(items):
    path = Path(f'output/{item}')
    path.mkdir(exist_ok=True, parents=True)
    try:
        pgs = PGSMoments.load(sorted(Path(args.shapesplat_ply).glob(f"{item}-*.ply"))[0])
    except:
        continue
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
    for ply in tqdm(plys, desc="rendering"):
        xyz, opacities, features_dc, scales, rots = load_ply_torch(ply)
        features_dc = SH2RGB(features_dc)

        gs_size = int(ply.stem.split('-')[-1])
        
        azi = azi % 360
        ele = ele % 360
        for i in range(6+int(np.log2(gs_size))*2):
            azi = azi + 10
            view_mat = np.linalg.inv(camera_matrix_from_angles(azi/180*np.pi, ele/180*np.pi, 1.4, np.array([0.0, 0.0, -1.0])))
            viewmats = torch.tensor(view_mat.tolist(), device='cuda')[None]

            output = gsplat.rasterization(xyz, rots, scales, opacities[:, 0], features_dc, viewmats, Ks, width, height)
        
            image = output[0][0]
            alpha = output[1][0]

            rgba = torch.cat([image, alpha], dim=-1)
            save_image(rgba.permute(2, 0, 1), ply.with_suffix(f'.{azi:03d}.{ele:03d}.png'))

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

    for fname in tqdm(images, desc='writing video'):
        fname = fname.as_posix()
        metas = fname.split('/')[-1].split('-')[-1]
        gs_size, azi, ele = metas.split('.')[:-1]
        azi = int(azi)
        ele = int(ele)
        frame = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if frame is not None:
            image, alpha = frame[:, :, :3], frame[:, :, 3:]
            # fill background with white color
            image = image + (255-alpha)
            image = cv2.putText(image, f'nums={gs_size}', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (56, 56, 56), 3, cv2.LINE_AA)
            image = cv2.putText(image, f'azi={azi}', (100, 180), cv2.FONT_HERSHEY_COMPLEX, 2, (56, 56, 56), 3, cv2.LINE_AA)
            image = cv2.putText(image, f'ele={ele}', (100, 260), cv2.FONT_HERSHEY_COMPLEX, 2, (56, 56, 56), 3, cv2.LINE_AA)

            video_writer.write(image)  # 写入帧
        else:
            print(f"警告: 无法读取图片 {fname}, 已跳过.")

    video_writer.release()
