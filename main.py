import sonata
import torch
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

split_head = torch.nn.Linear(1088, 1)
dense_head = torch.nn.Linear(1088, 2*14)
split_head.cuda()
dense_head.cuda()

opt = torch.optim.AdamW(list(model.parameters())+list(split_head.parameters())+list(dense_head.parameters()), lr=1e-4)

for epoch in range(1000):
    print(epoch)
    for level in range(output_back['level']-20):
        now_gs_index, next_gs_split, next_gs_index = output_back[level]

        now_gs = torch.from_numpy(output_back['data'][now_gs_index]).float()
        new_gs_gt = torch.from_numpy(output_back['data'][next_gs_index]).float().cuda()
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
        loss_mse = torch.nn.functional.mse_loss(dense_head(feat[next_gs_split]).view(-1, 2, 14), new_gs_gt)

        loss = loss_bce+loss_mse
        loss.backward()

        print(level, loss_bce.item(), loss_mse.item())

    opt.step()
    opt.zero_grad()
    