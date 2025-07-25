from .combine import GSItem, activated_gs_to_gs, activated_gs_to_train_gs
from .mix import mix_gaussian_inv

from tqdm import tqdm, trange
import kdtree
import numpy as np
import heapq

class PGS:
    def __init__(self, xyzs, opacities, features_dc, sigmas, inv_sigmas):
        gs_items = [
            GSItem(_xyz, _opacity, _features, _sigma, _inv_sigma, _index)
            for 
                _xyz, _opacity, _features, _sigma, _inv_sigma, _index 
            in 
                zip(xyzs, opacities, features_dc, sigmas, inv_sigmas, np.arange(xyzs.shape[0]))
        ]

        self.last_item_index = len(gs_items)
        self.valid_size = self.last_item_index
        self.valid_mask = [True] * self.valid_size

        self.tree = kdtree.create(gs_items)

        self.pq = gs_items
        heapq.heapify(self.pq)

    def simplify(self, max_size):
        merge_list = []
        
        if self.valid_size <= max_size: return merge_list

        tbar = trange(self.valid_size-max_size)
        while self.valid_size > max_size:
            item0 = heapq.heappop(self.pq)
            if not self.valid_mask[item0.index]: continue

            _item0, _item1 = self.tree.search_knn(item0, 2)
            
            self.valid_mask[_item0[0].data.index] = False
            self.valid_mask[_item1[0].data.index] = False

            o1 = _item0[0].data.opacity
            o2 = _item1[0].data.opacity

            u1 = _item0[0].data.xyz
            u2 = _item1[0].data.xyz

            f1 = _item0[0].data.features
            f2 = _item1[0].data.features

            s1 = _item0[0].data.Sigma
            s2 = _item1[0].data.Sigma

            inv_s1 = _item0[0].data.invSigma
            inv_s2 = _item1[0].data.invSigma

            u3, o3, f3, s3, inv_s3 = mix_gaussian_inv(u1, o1, f1, s1, inv_s1, u2, o2, f2, s2, inv_s2)
            
            new_item = GSItem(u3, o3, f3, s3, inv_s3, self.last_item_index)

            merge_list.append({
                "source": _item0[0].data,
                "target": _item1[0].data,
                "mixed": new_item,
            })

            heapq.heappush(self.pq, new_item)
            
            self.tree = self.tree.remove(_item0[0].data)
            self.tree = self.tree.remove(_item1[0].data)
            self.tree.add(new_item)

            self.last_item_index += 1
            self.valid_size -= 1
            self.valid_mask.append(True)

            tbar.update(1)

        return merge_list

    def to_gs_activated(self):
        """
        This function is used to convert the item to Tensor. \n
            1. If you want to get unactivated gs for saving, please use `to_gs`.
            2. If you want to get unactivated gs for training, please use `to_gs_activated`.            
        """
        new_xyz = np.zeros((self.valid_size, 3))
        new_opacities = np.zeros((self.valid_size, 1))
        new_features_dc = np.zeros((self.valid_size, 3))
        new_scales = np.zeros((self.valid_size, 3))
        new_rots = np.zeros((self.valid_size, 4))
        for i, node in enumerate(tqdm(list(self.tree.inorder()))):
            vector = node.data.vector()

            new_xyz[i] = vector[:3]
            new_opacities[i] = vector[3:4]
            new_features_dc[i] = vector[4:7]
            new_scales[i] = vector[7:10]
            new_rots[i] = vector[10:14]

        return new_xyz, new_opacities, new_features_dc, new_scales, new_rots

    def to_gs(self, delta=1e-10):
        """
        This function is used to convert the item to Gaussian Splatting format to exchange with other methods.
        """
        new_xyz, new_opacities, new_features_dc, new_scales, new_rots = self.to_gs_activated()

        new_xyz, new_opacities, new_features_dc, new_scales, new_rots = activated_gs_to_gs(new_xyz, new_opacities, new_features_dc, new_scales, new_rots, delta=delta)

        return new_xyz, new_opacities, new_features_dc, new_scales, new_rots
    
    def to_train_gs(self, delta=1e-10, beta=1.0):
        """
        This function replace the activation function of `scales` with `softplus` from `log`.
        """
        new_xyz, new_opacities, new_features_dc, new_scales, new_rots = self.to_gs_activated()

        new_xyz, new_opacities, new_features_dc, new_scales, new_rots = activated_gs_to_train_gs(new_xyz, new_opacities, new_features_dc, new_scales, new_rots, delta=delta, beta=beta)

        return new_xyz, new_opacities, new_features_dc, new_scales, new_rots