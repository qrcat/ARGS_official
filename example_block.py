import copy
from utils.io import activated_gs2gs, save_ply
from utils.quaternion import quaternion_multiply, normalize_quaternions
from pgs import PGSMoments, PGSMomentSample
from scipy.spatial.transform import Rotation
import numpy as np

pgs = PGSMoments.load("airplane-0.ply")
# pgs = PGSMomentSample.load("point_cloud.ply")
merge_list = pgs.simplify(1)
merge_list = merge_list[::-1]
# 建立分裂的连接关系
tmap = {}
for merge in merge_list:
    tmap[merge['mixed_id']] = [i if isinstance(i, int) else i.item() for i in [merge['source_id'], merge['target_id']]]

root = merge_list[0]['mixed_id']
# 获得分层的GS
level = 0
count = []
sequence, to_split, split_next, split_mask = [], [root], [], []
output = {}
while True:
    to_split_next = []

    count.append(len(to_split))

    for index in to_split:
        sequence.append(index)
        if tmap.get(index):
            split_next.append([tmap[index][0], tmap[index][1]])
            split_mask.append(True)

            to_split_next.append(tmap[index][0])
            to_split_next.append(tmap[index][1])
        else:
            split_next.append([index, index]) # 不分裂，填充自己的特征
            split_mask.append(False)
    
    if len(to_split_next) == 0: # 没有分裂的gs
        break

    to_split = to_split_next

cumsum = np.cumsum([0]+count[:-1])
mask_value = cumsum.repeat(count)

attn_mask = np.zeros((len(sequence), len(sequence))) # N, N
attn_mask[mask_value:] = True
# usage

mask_value = torch.from_numpy(mask_value).cuda()

# torch > 2.5.1
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= mask_value[kv_idx]

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=8192, KV_LEN=8192, BLOCK_SIZE=1, device="cuda")

attn_mask = block_mask.to_dense()
print(attn_mask[0, 0, :10, :10])
