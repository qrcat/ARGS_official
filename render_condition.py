from utils.render import camera_matrix_from_angles
from utils.shs import SH2RGB
from utils.io import  gs2activated_gs, load_ply
from torchvision.transforms.functional import normalize
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import open_clip
import argparse
import gsplat
import torch
import numpy as np
import time


def tensor_from_numpy(np_ls: list, device):
    return [torch.from_numpy(np_arr).to(device, dtype=torch.float) for np_arr in np_ls]

def load_ply_torch(path: str, device):
    rets = gs2activated_gs(*load_ply(path))
    return tensor_from_numpy(rets, device)

@torch.no_grad()
def worker(queue, count, device):
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    model.to(device)

    while True:
        item = queue.get()
        if item is None:
            break

        input_ply = item[0]

        width, height, focal = 224, 224, 200.0
        Ks = torch.tensor([[focal, 0.0, width/2.0], [0.0, focal, height/2.0], [0.0, 0.0, 1.0],], device=device)[None]

        xyz, opacities, features_dc, scales, rots = load_ply_torch(input_ply, device)

        features_dc = SH2RGB(features_dc)

        camera_pair_list = [
            (0, 30), (45, 30), (90, 30), (135, 30), (180, 30), (225, 30), (270, 30), (315, 30)
        ]

        cam_pos = []
        for azi, ele in camera_pair_list:
            radius   = xyz.norm(dim=1).max().item()*2
            view_mat = camera_matrix_from_angles(azi/180*np.pi, ele/180*np.pi, radius, np.array([0.0, 0.0, -1.0]))
            cam_pos.append(view_mat)
        viewmats = np.stack(cam_pos)
        viewmats = torch.from_numpy(viewmats).to(device, dtype=torch.float)

        B, _, _ = viewmats.shape
        # render image
        output = gsplat.rasterization(xyz, rots, scales, opacities[:, 0], features_dc, viewmats, Ks.repeat(B, 1, 1), width, height)
        image, alpha = output[:2]
        rgba = torch.cat([image, alpha], dim=-1)
        # breakpoint()
        for i in range(B):
            azi, ele = camera_pair_list[i]
            output_img = output_path / input_ply.parent.name / f'{input_ply.stem}_{azi}_{ele}.png'
            save_image(rgba[i].permute(2, 0, 1), output_img)
        # get dino features
        norm_image = normalize(image.permute(0, 3, 1, 2), [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

        clip_features = model.visual.forward_intermediates(
                norm_image,
                stop_early = False,
                normalize_intermediates = False,
                intermediates_only = True,
                output_fmt = 'NLC',
                output_extra_tokens = False,
        )['image_intermediates'][-1]

        output_clip = output_path / input_ply.parent.name / 'feature.npy'
        np.save(output_clip, clip_features.detach().cpu().numpy())
        
        with count.get_lock():
            count.value += 1

        tqdm.write(f"Processed {count.value} items")
        
        queue.task_done()

def main(args):
    global queue, count

    if args.type == 'modelsplat':
        pattern = '*/*.ply'

    input_plys = Path(args.input).glob(pattern)

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    processes = []

    # use parallel processing
    for worker_i in range(args.workers):
        process = multiprocessing.Process(
            target=worker,
            args=(queue, count, f'cuda:{worker_i}')
        )
        process.daemon = True
        process.start()
        processes.append(process)

    queue.join()
    
    for input_ply in tqdm(input_plys):
        queue.put((input_ply,))
    
    for worker_i in range(args.workers):
        queue.put(None)
        
    start_time = time.time()
    for p in processes:
        p.join()
    end_time = time.time()
    print(f"All tasks completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {count.value} items")

    print("All worker processes finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("--input", default="data/airplane_enhance")
    # outputs
    parser.add_argument("--output", default="data/airplance_pkl")
    # type
    parser.add_argument("--type", default="none", choices=['none', 'modelsplat', 'shapesplat'])
    # workers
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    queue, count = None, None
    output_path  = Path(args.output)

    main(args)
