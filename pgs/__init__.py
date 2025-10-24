# from .merge import merge_gaussian, merge_gaussian_inv, merge_gaussian_moments, merge_gaussian_moments_ub
import pgs.merge as merge
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
        self._index = np.arange(self.full_size)

        self._dist_w = np.array([100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0])

        # update pq
        self.pq = [(_size, _opa, _index) for _size, _opa, _index in zip(np.prod(self.scales, axis=-1), self.opacities, self.index)]
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

    def simplify(self, num_points, merge_method='merge_gaussian_moments'):
        merge_list = []

        if self.used_size <= num_points: return merge_list

        tbar = trange(self.used_size-num_points, leave=False)
        while self.used_size > num_points:
            head = heapq.heappop(self.pq)
            _, _, index_1 = head
            if not self.valid_mask[index_1]: continue
            # find neighbor
            index = faiss.IndexFlatL2(14)
            index.add(self.weighted_data[self.valid_mask])

            query = self.weighted_data[index_1]

            _, indices = index.search(query[None], 2) # this is local indices of self.valid_mask
            
            index_2 = self.index[self.valid_mask][indices[0][1]]

            # remove the node
            self.valid_mask[index_1] = False
            self.valid_mask[index_2] = False
            
            # ==============================================================================================
            # get the original gaussian
            xyz_1, xyz_2 = self.xyz_global[index_1], self.xyz_global[index_2]
            opa_1, opa_2 = self.opacities_global[index_1], self.opacities_global[index_2]
            rgb_1, rgb_2 = self.features_global[index_1], self.features_global[index_2]
            scl_1, scl_2 = self.scales_global[index_1], self.scales_global[index_2]
            qut_1, qut_2 = self.quats_global[index_1], self.quats_global[index_2]

            sigma_1, inv_sigma_1 = build_sigma(scl_1, qut_1)
            sigma_2, inv_sigma_2 = build_sigma(scl_2, qut_2)
            
            xyz_3, opa_3, rgb_3, sigma_3, inv_sigma_3 = getattr(merge, merge_method)(
                xyz_1, opa_1, rgb_1, sigma_1, inv_sigma_1, 
                xyz_2, opa_2, rgb_2, sigma_2, inv_sigma_2, 
                cross=True
            )
            scl_3, qut_3 = unpack_sigma(sigma_3)
            qut_3 = norm_quats(qut_3)

            # update the gaussian
            index_3 = self.last_index # global index

            self.used_size -= 1 # update index

            self._data[index_3] = np.concatenate([xyz_3, opa_3, rgb_3, scl_3, qut_3])
            self.valid_mask[index_3] = True

            merge_list.append({
                "source": np.concatenate([xyz_1, opa_1, rgb_1, scl_1, qut_1]),
                "source_id": index_1 if isinstance(index_1, int) else index_1.item(),
                "target": np.concatenate([xyz_2, opa_2, rgb_2, scl_2, qut_2]),
                "target_id": index_2 if isinstance(index_2, int) else index_2.item(),
                "mixed": np.concatenate([xyz_3, opa_3, rgb_3, scl_3, qut_3]),
                "mixed_id": index_3,
            })

            heapq.heappush(self.pq, (np.prod(self.scales_global[index_3]), self.opacities_global[index_3], index_3))

            tbar.update(1)
        return merge_list

    @staticmethod
    def load(path, save=False, debug=False) -> 'PGSMoments':
        path = Path(path)

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

        return pgs

class PGSMomentSample(PGSMoments):
    def __init__(self, xyz, opacities, features, scales, quats):

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

    def simplify(self, num_points):
        # breakpoint()
        merge_list = []

        if self.used_size <= num_points: return merge_list

        tbar = trange(self.used_size-num_points, leave=False)
        while self.used_size > num_points:
            volume = np.prod(self.scales, axis=-1)
            candidate_index = np.argpartition(volume, -min(10, len(volume)))[-10:]
            candidate_probs = -np.log(volume[candidate_index].clip(max=1))
            candidate_probs = candidate_probs / candidate_probs.sum()
            _index1 = np.random.choice(candidate_index, 1, p=candidate_probs)
            # breakpoint()
            _index1 = self.index[self.valid_mask][_index1]

            index = faiss.IndexFlatL2(14)
            index.add(self.weighted_data[self.valid_mask])

            _query = self.weighted_data[_index1]
            
            distance, indices = index.search(_query, 2) # this is local indices
            _index2 = self.index[self.valid_mask][indices[0, 1]]
            _index1 = _index1[0]

            # check if the edge is valid
            if not self.valid_mask[_index1] or not self.valid_mask[_index2]: continue
            # remove the edge
            self.valid_mask[_index1] = False
            self.valid_mask[_index2] = False
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
            alpha = 1.0
            _xyz3, _opacity3, _feature3, _sigma3, _inv_sigma3 = merge_gaussian_moments_ub(
                _xyz1, _opacity1, _feature1, _sigma1, _inv_sigma1, 
                _xyz2, _opacity2, _feature2, _sigma2, _inv_sigma2, 
                cross=True, alpha=alpha,
            )
            _scale3, _quat3 = unpack_sigma(_sigma3)
            _quat3 = norm_quats(_quat3)

            # update the gaussian
            _index3 = self.last_index

            self.used_size -= 1 # update index

            self._data[_index3] = np.concatenate([_xyz3, _opacity3, _feature3, _scale3, _quat3])
            self.valid_mask[_index3] = True

            merge_list.append({
                "source": np.concatenate([_xyz1, _opacity1, _feature1, _scale1, _quat1]),
                "source_id": _index1.item(),
                "target": np.concatenate([_xyz2, _opacity2, _feature2, _scale2, _quat2]),
                "target_id": _index2.item(),
                "mixed": np.concatenate([_xyz3, _opacity3, _feature3, _scale3, _quat3]),
                "mixed_id": _index3,
            })

            tbar.update(1)
        return merge_list
    @staticmethod
    def load(path, save=False, debug=False) -> 'PGSMoments':
        path = Path(path)

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

        pgs = PGSMomentSample(xyz, opacities, features_dc, scales, rots)

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
