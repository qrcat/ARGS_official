from utils.quaternion import quaternion_multiply, quaternion_inverse, normalize_quaternions
import sonata
import torch
import torch.nn as nn
import numpy as np



config = [
    # dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    # dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "inverse"),
        feat_keys=("coord", "color", "sigma"),
    ),
]
transform = sonata.transform.Compose(config)

custom_config = dict(
    enc_patch_size=[1024 for _ in range(5)],
    enable_flash=False,  # reduce patch size if necessary
)
model = sonata.load("sonata", repo_id="facebook/sonata", custom_config=custom_config).cuda()
model.embedding.stem.linear = torch.nn.Linear(14, out_features=48, bias=True)
model.embedding.stem.linear.cuda()

import pickle
# 读取这个文件
with open('data.pkl', 'rb') as f:
    output_back = pickle.load(f)

split_head = nn.Sequential(nn.Linear(1088, 512), nn.LeakyReLU(), nn.Linear(512, 1))
dense_head = nn.Sequential(nn.Linear(1088, 512), nn.LeakyReLU(), nn.Linear(512, 2*14))
split_head.cuda()
dense_head.cuda()

opt = torch.optim.AdamW([
    {
        'params': model.parameters(),
        'lr': 1e-5,
    },
    {
        'params': split_head.parameters(),
        'lr': 1e-3,
    },
    {
        'params': dense_head.parameters(),
        'lr': 1e-3,
    },
])

# ===========================
# Local/Global transforms
# ===========================
def to_local(parent: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
    """
    parent:   (M, 14)
    children: (M, 2, 14)
    return:   (M, 2, 14)  order:
      Δpos(3), Δopacity(ratio)(1), Δfeat(3), Δscale(log ratio)(3), Δquat(relative)(4)
    """
    eps = 1e-6
    local = torch.zeros_like(children)

    # Δpos
    local[..., :3] = children[..., :3] - parent[:, None, :3]
    # Δopacity (ratio)
    p_op = torch.clamp(parent[:, 3:4], min=eps)
    c_op = torch.clamp(children[..., 3:4], min=eps)
    local[..., 3:4] = c_op / p_op[:, None, :]
    # Δfeature (difference)
    local[..., 4:7] = children[..., 4:7] - parent[:, None, 4:7]
    # Δscale = log(child/parent)
    p_s = torch.clamp(parent[:, 7:10], min=eps)
    c_s = torch.clamp(children[..., 7:10], min=eps)
    local[..., 7:10] = torch.log(c_s / p_s[:, None, :])
    # Δquat = q_parent^{-1} ⊗ q_child
    q_parent = normalize_quaternions(parent[:, -4:])
    q_child0 = normalize_quaternions(children[:, 0, -4:])
    q_child1 = normalize_quaternions(children[:, 1, -4:])
    q_local0 = quaternion_multiply(quaternion_inverse(q_parent), q_child0)
    q_local1 = quaternion_multiply(quaternion_inverse(q_parent), q_child1)
    local[:, 0, -4:] = normalize_quaternions(q_local0)
    local[:, 1, -4:] = normalize_quaternions(q_local1)
    return local

for epoch in range(10000):
    print(epoch)
    for level in range(output_back['level']-20):
        now_gs_index, next_gs_split, next_gs_index = output_back[level]

        now_gs = torch.from_numpy(output_back['data'][now_gs_index]).float()
        new_gs_gt = torch.from_numpy(output_back['data'][next_gs_index]).float()
        new_gs = to_local(now_gs[next_gs_split], new_gs_gt)
        # breakpoint()
        point = {
            "coord": np.array(now_gs[...,  :3]),  # (N, 3) 3postion
            "color": np.array(now_gs[..., 3:7]),  # (N, 4) 1opacity + 3color
            "sigma": np.array(now_gs[..., 7: ]),  # (N, 7) 3scale   + 4scale
        }

        point = transform(point)
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)

        point = model(point)
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        feat = point.feat[point.inverse]

        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(split_head(feat), torch.tensor(next_gs_split).float().cuda()[..., None])
        loss_mse = torch.nn.functional.mse_loss(dense_head(feat[next_gs_split]).view(-1, 2, 14), new_gs.cuda())

        loss = loss_bce+loss_mse
        loss.backward()

        print('{:02d} d{:.02f}% bce:{:.03f} mse:{:.03f}'.format(level, np.sum(next_gs_split)/len(next_gs_split)*100, loss_bce.item(), loss_mse.item()))

    opt.step()
    opt.zero_grad()
    