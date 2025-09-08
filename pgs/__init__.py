from .merge import merge_gaussian, merge_gaussian_inv, merge_gaussian_moments
from utils.io import get_combinable_gaussian, activated_gs2gs, activated_gs2train_gs, gs2activated_gs, save_ply, load_ply
from utils.quantize import Quantize
from utils.gaussian import build_sigma, unpack_sigma, norm_quats

from typing import List
from pathlib import Path
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R
# import kdtree
import faiss
import os
import numpy as np
import heapq
import pickle


class ARGS(object):
    def __init__(self):
        pass
    
    @property
    def xyz(self):
        raise NotImplementedError

    @property
    def opacities(self):
        raise NotImplementedError
    
    @property
    def features(self):
        raise NotImplementedError
    
    @property
    def scales(self):
        raise NotImplementedError
    
    @property
    def quats(self):
        raise NotImplementedError

    def get(self):
        """
        This function is used for getting the data.
        """
        return self.xyz, self.opacities, self.features, self.scales, self.quats


class ProgressiveGaussianSimplifierBase(ARGS):
    def __init__(self):
        self.full_size = None
        self.used_size = None
        
        self.quantize = Quantize()

    @property
    def last_index(self):
        return self.full_size - self.used_size + 1

    @staticmethod
    def load(path, save=True) -> 'ProgressiveGaussianSimplifierBase':
        raise NotImplementedError

    def simplify(self, num_points) -> List:
        raise NotImplementedError

    def to_gs(self, delta=1e-10):
        """
        This function is used to convert the item to Gaussian Splatting format to exchange with other methods.
        """
        new_xyz, new_opacities, new_features, new_scales, new_rots = self.get()

        new_xyz, new_opacities, new_features, new_scales, new_rots = activated_gs2gs(new_xyz, new_opacities, new_features, new_scales, new_rots, delta=delta)

        return new_xyz, new_opacities, new_features, new_scales, new_rots

    def save(self, path):
        new_xyz, new_opacities, new_features_dc, new_scales, new_rots = self.to_gs()
        save_ply(path, new_xyz, new_opacities, new_features_dc, new_scales, new_rots)



class PGSMoments(ProgressiveGaussianSimplifierBase):
    def __init__(self, xyz, opacities, features, scales, quats):
        super().__init__()

        self.used_size = len(xyz)
        self.full_size = 2*self.used_size-1

        self._data = np.zeros((self.full_size, 14))
        # convert to sorted order because `unpack_sigma` use SVD to sort the scales
        sigma, inv_sigma = build_sigma(scales, quats)
        scales, quats = unpack_sigma(sigma)
        # attach the attributes
        self._data[:self.used_size, :3] = xyz
        self._data[:self.used_size, 3:4] = opacities
        self._data[:self.used_size, 4:7] = features
        self._data[:self.used_size, 7:10] = scales
        self._data[:self.used_size, 10:] = norm_quats(quats)
        # convert to float64 for digital stability
        self._data = self._data.astype(np.float64)
        # aux attributions
        self._valid_mask = np.asarray([True] * self.used_size + [False] * (self.used_size-1))
        # update the neighborhood
        dists = np.zeros(self.used_size)
        self._index = np.arange(self.full_size)
        self._neighbor = np.full(self.full_size, -1, dtype=np.int32)
        
        self._dist_w = np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])

        index = faiss.IndexFlatL2(14)
        index.add(self.weighted_data)
        for _i, _q in enumerate(tqdm(self.weighted_data, leave=False)):
            distance, indices = index.search(_q[None], 2)
            dists[_i] = distance[0, 1]
            self._neighbor[_i] = indices[0, 1]
        # update pq
        self.pq = [(_size, _dist, _index, _neighbor) for _size, _dist, _index, _neighbor in zip(np.prod(self.scales, axis=-1), dists, self.index, self.neighbor)]
        heapq.heapify(self.pq)

    @property
    def weighted_data(self):
        return self._data[:self.last_index] * self._dist_w
    # for get current gs
    # =========================================================================
    @property
    def xyz(self):
        return self._data[self._valid_mask, :3]
    
    @property
    def opacities(self):
        return self._data[self._valid_mask, 3:4]
    
    @property
    def features(self):
        return self._data[self._valid_mask, 4:7]

    @property
    def scales(self):
        return self._data[self._valid_mask, 7:10]

    @property
    def quats(self):
        return self._data[self._valid_mask, 10:]
    # =========================================================================

    @property
    def index(self):
        return self._index[:self.last_index]

    @property
    def neighbor(self):
        return self._neighbor[:self.last_index]

    @property
    def valid_mask(self):
        return self._valid_mask[:self.last_index]


    @property
    def neighbor(self):
        return self._neighbor[:self.last_index]
    
    @property
    def xyz_global(self):
        return self._data[:self.last_index, :3]
    
    @property
    def opacities_global(self):
        return self._data[:self.last_index, 3:4]
    
    @property
    def features_global(self):
        return self._data[:self.last_index, 4:7]

    @property
    def scales_global(self):
        return self._data[:self.last_index, 7:10]

    @property
    def quats_global(self):
        return self._data[:self.last_index, 10:]

    def update_neighbor(self, _index):
        index = faiss.IndexFlatL2(14)
        index.add(self.weighted_data[self.valid_mask])

        _query = self.weighted_data[_index]
        
        distance, indices = index.search(_query, 2) # this is local indices

        for index, dist, neighbor in zip(_index, distance, self.index[self.valid_mask][indices]):
            if not self.valid_mask[index]: 
                self.neighbor[index] = -1
                continue
            # print(index, dist, neighbor, self.valid_mask[index])
            self.neighbor[index] = neighbor[1]

            # update heapq
            heapq.heappush(self.pq, (np.prod(self.scales_global[index]), dist[1], index, neighbor[1]))

    def simplify(self, num_points):
        merge_list = []

        if self.used_size <= num_points: return merge_list

        tbar = trange(self.used_size-num_points, leave=False)
        while self.used_size > num_points:
            head = heapq.heappop(self.pq)
            _size, _dist, _index1, _index2 = head
            # check if the edge is valid
            if not self.valid_mask[_index1] or not self.valid_mask[_index2]: continue
            # remove the edge
            self.valid_mask[_index1] = False
            self.valid_mask[_index2] = False
            # for those who reference _index1 and _index2 are invalid
            select_neighbor = np.bitwise_or(self.neighbor == _index1, self.neighbor == _index2)
            _index = self.index[select_neighbor]
            _neighbor = self.neighbor[select_neighbor]
            # ==============================================================================================
            # get the original gaussian
            _xyz1, _xyz2 = self.xyz_global[_index1], self.xyz_global[_index2]
            _opacity1, _opacity2 = self.opacities_global[_index1], self.opacities_global[_index2]
            _feature1, _feature2 = self.features_global[_index1], self.features_global[_index2]
            _scale1, _scale2 = self.scales_global[_index1], self.scales_global[_index2]
            _quat1, _quat2 = self.quats_global[_index1], self.quats_global[_index2]

            _sigma1, _inv_sigma1 = build_sigma(_scale1, _quat1)
            _sigma2, _inv_sigma2 = build_sigma(_scale2, _quat2)
            
            # compute the merged gaussian
            alpha = 0.4 * min(np.log2(self.used_size)/16, 1.0) + 0.6
            _xyz3, _opacity3, _feature3, _sigma3, _inv_sigma3 = merge_gaussian_moments(
                _xyz1, _opacity1, _feature1, _sigma1, _inv_sigma1, 
                _xyz2, _opacity2, _feature2, _sigma2, _inv_sigma2, 
                cross=True, alpha=alpha,
            )
            _scale3, _quat3 = unpack_sigma(_sigma3)
            _quat3 = norm_quats(_quat3)

            # # modify opacity and scales (by power)
            # # ============================================================
            # _opacity3_modify = _opacity1 + _opacity2 - _opacity1*_opacity2
            # _opacity3_scaled = _opacity3_modify**0.5 / _opacity3**0.5
            # _scale3_modify = _scale3 / _opacity3_scaled**(1/3)
            # _opacity3, _scale3 = _opacity3_modify, _scale3_modify
            # # ============================================================

            # update the gaussian
            _index3 = self.last_index

            self.used_size -= 1 # update index

            self._data[_index3] = np.concatenate([_xyz3, _opacity3, _feature3, _scale3, _quat3])
            self.valid_mask[_index3] = True

            merge_list.append({
                "source": np.concatenate([_xyz1, _opacity1, _feature1, _scale1, _quat1]),
                "target": np.concatenate([_xyz2, _opacity2, _feature2, _scale2, _quat2]),
                "mixed": np.concatenate([_xyz3, _opacity3, _feature3, _scale3, _quat3]),
            })

            update_index_set = set([_index3] + _index.tolist() + _neighbor.tolist())
            self.update_neighbor(list(update_index_set))
            
            tbar.update(1)
        return merge_list

    @staticmethod
    def load(path, save=False, debug=False) -> 'PGSMoments':
        path = Path(path)
        pkl = path.with_suffix('.pkl')
        if os.path.exists(pkl):
            with open(pkl, 'rb') as f:
                pgs = pickle.load(f)
        else:
            xyz, opacities, features_dc, scales, rots = gs2activated_gs(*load_ply(path))
            # normalize the data
            center = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
            xyz = xyz - center
            scale = (xyz.max(axis=0) - xyz.min(axis=0)).max()
            xyz = xyz / scale
            scales = scales / scale

            if debug:
                xyz, opacities, features_dc, scales, rots = activated_gs2gs(xyz, opacities, features_dc, scales, rots)
                save_ply('test.ply', xyz, opacities, features_dc, scales, rots)
                exit()

            pgs = PGSMoments(xyz, opacities, features_dc, scales, rots)

            if save:
                with open(pkl, 'wb') as f: pickle.dump(pgs, f)
        return pgs

class PGSTreeNode:
    def __init__(self, xyz, sigma, index):
        self.xyz = xyz
        self.sigma = sigma
        self.index = index
    
    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, index):
        return self.xyz[index]

    def __repr__(self):
        return f'PGSTreeNode({self.xyz}, {np.linalg.det(self.sigma)},{self.index})'

    def __lt__(self, other):
        return np.linalg.det(self.sigma) < np.linalg.det(other.sigma)
