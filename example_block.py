import copy
from utils.io import activated_gs2gs, save_ply
from utils.quaternion import quaternion_multiply, normalize_quaternions
from pgs import PGSMoments, PGSMomentSample
from scipy.spatial.transform import Rotation
import numpy as np

pgs = PGSMoments.load("point_cloud.ply")
# pgs = PGSMomentSample.load("point_cloud.ply")
merge_list = pgs.simplify(1)
merge_list = merge_list[::-1]
# 建立分裂的连接关系
tmap = {}
for merge in merge_list:
    tmap[merge['mixed_id']] = [i if isinstance(i, int) else i.item() for i in [merge['source_id'], merge['target_id']]]

root = merge_list[0]['mixed_id']
prev_gs_to_split = [root]
count = []
sequence, split_gs, split_bl = [], [], []
output = {}
while True:
    next_gs_to_split = []

    count.append(len(prev_gs_to_split))

    for index in prev_gs_to_split:
        sequence.append(index)
        if tmap.get(index):
            split_gs.append([tmap[index][0], tmap[index][1]])
            split_bl.append(True)

            next_gs_to_split.append(tmap[index][0])
            next_gs_to_split.append(tmap[index][1])
        else:
            split_gs.append([index, index]) # 不分裂，填充自己的特征
            split_bl.append(False)
    
    if len(next_gs_to_split) == 0: # 没有分裂的gs
        break

    prev_gs_to_split = next_gs_to_split

cumsum = np.cumsum([0]+count[:-1])
mask_value = cumsum.repeat(count)

output['data'] = pgs._data
output['count'] = np.array(count)
output['cumsum'] = cumsum
output['sequence'] = np.array(sequence)
output['split_gs'] = np.array(split_gs)
output['split_bl'] = np.array(split_bl)


import pickle
with open('data_block.pkl', 'wb') as f:
    pickle.dump(output, f)
# usage
import torch
mask_value = torch.from_numpy(mask_value).cuda()

# torch > 2.5.1
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= mask_value[kv_idx]

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=8192, KV_LEN=8192, BLOCK_SIZE=1, device="cuda")

attn_mask = block_mask.to_dense()
print(attn_mask[0, 0, :10, :10])
