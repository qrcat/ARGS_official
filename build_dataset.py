
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


def get_balance_group(coords, n_groups=8, group_items=256, seed=42, debug=False):
    assert coords.shape[0]==group_items*n_groups
    np.random.seed(seed)

    # 步骤1：K-means聚类（按空间坐标聚为n_groups类）
    kmeans = KMeans(n_clusters=n_groups, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(coords)  # 每个高斯的聚类标签

    # 步骤2：统计初始聚类的数量，准备微调
    group_counts = defaultdict(int)
    for label in labels:
        group_counts[label] += 1
    
    if debug: print("init cluster number", [group_counts[i] for i in range(5)])

    # 步骤3：移补法平衡每组至512个
    # 1. 找出超量组（>512）和欠量组（<512）
    over_groups = [g for g in range(n_groups) if group_counts[g] > group_items]
    under_groups = [g for g in range(n_groups) if group_counts[g] < group_items]

    # 2. 为每个超量组，选出“边缘样本”（距离聚类中心最远的样本）移至欠量组
    final_labels = labels.copy()
    for g_over in over_groups:
        # 超量组的样本索引
        over_indices = np.where(labels == g_over)[0]
        # 计算样本到聚类中心的距离（距离越远越边缘）
        center = kmeans.cluster_centers_[g_over]
        distances = np.linalg.norm(coords[over_indices] - center, axis=1)
        # 按距离排序，取超量的样本（需移出的样本）
        n_remove = group_counts[g_over] - group_items
        remove_indices = over_indices[np.argsort(distances)[-n_remove:]]  # 最远的n_remove个
        
        # 将移出的样本分配给欠量组
        for idx in remove_indices:
            # 选一个欠量组
            arg = np.argmin([np.linalg.norm(coords[idx] - kmeans.cluster_centers_[g_under]) for g_under in under_groups])
            g_under = under_groups[arg]
            # g_under = under_groups[0]
            final_labels[idx] = g_under
            # 更新欠量组计数，满了就移除
            group_counts[g_under] += 1
            if group_counts[g_under] == group_items:
                under_groups.pop(arg)
    
    # 生成最终分组
    groups = [np.where(final_labels == g)[0] for g in range(n_groups)]

    if debug: print("balanced cluster number", [len(g) for g in groups][:5])  # 输出：[512, 512, 512, 512, 512]

    return groups
cluster_orders = [512, 64, 8, 1]
cluster_params = {
    512: 64,
    64: 128,
    8: 256,
    1: 512
}
import json

cluster_meta = {
   "seqs": [cluster_params[order] for order in cluster_orders],
   "order": cluster_orders,
}

with open("datasets/cluster_meta.json", "w") as f:
   json.dump(cluster_meta, f, indent=4)

with open("datasets/cluster_meta.json") as f:
    cluster_meta = json.load(f)

scale_gaussian = {}
for level, n_groups in enumerate(cluster_orders):
    group_items = cluster_params[n_groups]
    gaussian_number = group_items * n_groups

    merge_list = pgs.simplify(gaussian_number)

    new_xyz, new_opacities, new_features_dc, new_scales, new_rots = pgs.to_train_gs(delta=1e-10)

    coords = np.concatenate(
        [new_xyz, new_opacities, new_features_dc, new_scales, new_rots], 
        axis=-1
    )

    groups = get_balance_group(coords, n_groups, coords.shape[0]//n_groups)

    sorted_source = []
    for i in range(n_groups):
        sub_source = coords[groups[i]]
        sub_scales = softplus_np(sub_source[:, 7:10], beta=1.0)
        reindex = np.argsort(sub_scales[:, 0] * sub_scales[:, 1] * sub_scales[:, 2])
        sub_source = sub_source[reindex[::-1]]

        sorted_source.append(sub_source)
        
    sorted_source = np.stack(sorted_source)

    scale_gaussian[f"scale_{level}"] = sorted_source
    print(f"generate scale {level}")

np.savez('/mnt/e/code/ARGS/datasets/0/0.npz', **scale_gaussian)
