import copy
from utils.io import activated_gs2gs, save_ply
from pgs import PGSMoments
import numpy as np

pgs = PGSMoments.load("point_cloud.ply")
merge_list = pgs.simplify(1)
merge_list = merge_list[::-1]
# 建立分裂的连接关系
tmap = {}
for merge in merge_list:
    tmap[merge['mixed_id']] = [merge['source_id'], merge['target_id']]
root = merge_list[0]['mixed_id']
# 获得分层的GS
level = 0
output = {}
now_gs_index = [root]
while True:
    next_gs_index = []
    next_gs_split = []
    for index in now_gs_index:
        if tmap.get(index):
            next_gs_index.append(tmap[index][0])
            next_gs_index.append(tmap[index][1])
            next_gs_split.append(True)
        else:
            next_gs_index.append(index)
            next_gs_split.append(False)
    if sum(next_gs_split) == 0: # 没有分裂的gs
        break
    output[level] = (now_gs_index, next_gs_split, [tmap[index] for index in now_gs_index if tmap.get(index)])
    now_gs_index = copy.deepcopy(next_gs_index)

    level += 1
# 添加GS数据
output['data'] = pgs._data.copy()
output['level'] = level # indice: 0~level-1

# 我们要保存的文件
import pickle
with open('data.pkl', 'wb') as f:
    pickle.dump(output, f)
# 读取这个文件
with open('data.pkl', 'rb') as f:
    output_back = pickle.load(f)

# ===================================================================================================
# = 训练时                                                                                          =
# ===================================================================================================
import torch
import torch.nn as nn

pointnet = ... # 使用pointnet提取特征
split_head = nn.Linear(..., 1) # 获得分裂结果
dense_head = nn.Linear(..., 2*14) # 获得两个生成的高斯
loss_func = ... # 监督生成高斯的损失函数


level = np.random.randint(0, output_back['level']) # 随机选一层
now_gs_index, next_gs_split, next_gs_index = output_back[level]

now_gs = torch.from_numpy(output_back['data'][now_gs_index]).float() # B, 14

# 保存下来可视化
now_gs_save = activated_gs2gs(now_gs)
save_ply('test.ply', now_gs_save[..., :3], now_gs_save[..., 3:4], now_gs_save[..., 4:7], now_gs_save[..., 7:10], now_gs_save[..., 10:14])

# 根据now_gs获得特征
features = pointnet(now_gs) # B, C

# 分类损失函数
torch.nn.functional.binary_cross_entropy_with_logits(split_head(features), next_gs_split.float())
# 生成高斯监督
now_gs_split = now_gs[next_gs_split].clone() # M, 14
new_gs_gt = torch.from_numpy(output_back['data'][next_gs_index]).clone() # M, 2, 14
# 计算新gs的local特征
new_gs_local = torch.zeros_like(new_gs_gt)
...# x,y,z  ->Δx,Δy,Δz                                              | M, 2, 3
new_gs_local[..., :3] = new_gs_gt[..., :3] - now_gs_split[..., None, :3] 
...# opacity->Δopacity                                              | M, 2, 1
...# feature->Δfeature                                              | M, 2, 3
...# scale  ->Δscale                                                | M, 2, 3
...# quat   ->Δquat                                                 | M, 2, 4

new_gs_pred = dense_head(features[next_gs_split]).view(-1, 2, 14) # 获得要分裂的点
loss_func(new_gs_pred, new_gs_local) # M, 2, 14
# ===================================================================================================
# = 推理时                                                                                          =
# ===================================================================================================
# 根据now_gs获得特征
features = pointnet(now_gs) # N, C

# 获得分裂gs的mask
split_mask = split_head(features) # N, 1
# 获得分裂gs
now_gs_split = now_gs[torch.squeeze(split_mask)>0].clone() # M', 14
# 获得分裂后的gs
new_gs_pred = dense_head(features[next_gs_split]).view(-1, 2, 14) # M', 2, 14
# 将这些local的gs变成global的
new_gs_pred_global = torch.zeros_like(new_gs_pred)
...# Δx,Δy,Δz->x,y,z                                                | M, 2, 3
new_gs_pred_global[..., :3] = now_gs_split[..., None, :3] + new_gs_pred[..., :3]
...# Δopacity->opacity                                              | M, 2, 1
...# ...

# 不保留那些分裂的gs
now_gs = now_gs[~torch.squeeze(split_mask)>0]
# 将分裂gs展平
new_gs_pred_global = new_gs_pred_global.view(-1, 14) # 2xM', 14
# 拼接
now_gs = torch.cat([now_gs, new_gs_pred_global], dim=0)
