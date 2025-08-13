from .io import unpack_gaussian_parameters, train_gs2activated_gs, activated_gs2gs, save_ply
from .gaussian import build_sigma, unpack_sigma, norm_quats
from .quantize import Quantize
from pgs import ARGS
from pgs.merge import merge_gaussian_inv, merge_gaussian_moments
from math import log2
import torch


class ARDecoder(ARGS):
    def __init__(self, max_point=16384, device='cuda'):

        self.quantize = Quantize()

        self._data = torch.zeros((max_point, 14), device=device)
        self._index = torch.arange(max_point, device=device)

        self._valid_mask = torch.zeros((max_point,), dtype=torch.bool)
        self._item_index = 0

        self.dist_weight = torch.tensor(
            [
                10.0, 10.0, 10.0, 
                1.0, 
                1.0, 1.0, 1.0,
                10.0, 10.0, 10.0, 
                1.0, 1.0, 1.0, 1.0
            ], device=device
        )

    @property
    def data(self):
        return self._data[self._valid_mask]

    @property
    def weighted_data(self):
        return self.data * self.dist_weight

    @property
    def xyz(self):
        return self._data[self._valid_mask, 0:3]
    
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
        return self._data[self._valid_mask, 10:14]
    
    @property
    def index(self):
        return self._index[self._valid_mask]

    def __len__(self):
        if self._item_index == 0:
            return 0
        else:
            return self._item_index // 2 + 1

    @torch.no_grad()
    def update(self, gaussian, top_k=128, solve_by='scale'):
        gaussian = self.quantize(gaussian)

        if self._item_index > 0:
            xyzs, opacities, features, scales, quats = unpack_gaussian_parameters(gaussian)

            sigma, inv_sigma = build_sigma(scales, quats)
            alpha = 0.4 * min(log2(len(self)+2)/16, 1.0) + 0.6
            xyz3, opacity3, feature3, sigma3, inv_sigma3 = merge_gaussian_moments(
                xyzs[0], opacities[0], features[0], sigma[0], inv_sigma[0],
                xyzs[1], opacities[1], features[1], sigma[1], inv_sigma[1],
                cross=True, alpha=alpha
            )
            scale3, quat3 = unpack_sigma(sigma3)

            new_features = torch.cat([xyz3, opacity3, feature3, scale3, quat3], dim=-1)
            
            # we got k-items and select the max one to remove
            if solve_by == 'scale':
                _size = torch.prod(self.scales, dim=-1)
                _, _indices = torch.topk(_size, k=min(top_k, len(self)))
                _dist = torch.norm(new_features*self.dist_weight - self.weighted_data[_indices], dim=-1)
                _select = _indices[torch.argmin(_dist)]
            elif solve_by == 'dist':
                _dist = torch.norm(new_features*self.dist_weight - self.weighted_data, dim=-1)
                _, _indices = torch.topk(-_dist, k=min(top_k, len(self)))
                _size = torch.prod(self.scales[_indices], dim=-1)
                _select = _indices[torch.argmax(_size)]
            else:
                _dist = torch.norm(new_features*self.dist_weight - self.weighted_data, dim=-1)
                _select = torch.argmin(_dist)


            self._valid_mask[self.index[_select]] = False
        # add new gs
        self._data[self._item_index:self._item_index+2] = gaussian
        self._valid_mask[self._item_index:self._item_index+2] = True
        self._item_index += 2
    
    def save_ply(self, path: str) -> None:
        save_ply(path, *activated_gs2gs(*self.get()))
    


# class ARDecoder(ARGS):
#     def __init__(self, max_point=16384, device='cuda'):


#         self.xyz = torch.zeros((max_point, 3), device=device)
#         self.opacity = torch.zeros((max_point, 1), device=device)
#         self.features = torch.zeros((max_point, 3), device=device)
#         self.scales = torch.zeros((max_point, 3), device=device)
#         self.quats = torch.zeros((max_point, 4), device=device)
#         self.sigma = torch.zeros((max_point, 3, 3), device=device)
#         self.inv_sigma = torch.zeros((max_point, 3, 3), device=device)

#         self.index = torch.arange(max_point, device=device)

#         self.tree = None

#         self.valid_size = 0
#         self.valid_mask = torch.zeros((max_point,), dtype=torch.bool)

#         self.dist_weight = torch.tensor(
#             [
#                 10.0, 10.0, 10.0, 
#                 1.0, 
#                 1.0, 1.0, 1.0,
#                 10.0, 10.0, 10.0, 
#                 1.0, 1.0, 1.0, 1.0
#             ], device=device
#         )

#         self.pq = []

#     @torch.no_grad()
#     def quantize(self, xyz, opacities, features_dc, scales, quats):
#         space_grid_nums = 2**11
#         xyz_clamped = xyz.clip(-0.5, 0.5)
#         xyz_indices = torch.round((xyz_clamped + 0.5) * space_grid_nums)
#         quantized_xyz = -0.5 + xyz_indices / space_grid_nums

#         quantized_opacity = torch.round(opacities * 255) / 255

#         C0 = 0.28209479177387814
#         quantized_feature = torch.round((features_dc + 0.5) * C0 * 255) / 255 / C0 - 0.5

#         min_scale = 2**-16
#         max_scale = 1.0
#         scale_grid_nums = 2**10
#         scale_clamped = scales.clip(min_scale, max_scale)
#         scale_indices = torch.round((torch.log2(scale_clamped) - log2(min_scale)) / (-log2(min_scale)) * scale_grid_nums)
#         quantized_scale = torch.exp2(scale_indices * (-log2(min_scale)) / scale_grid_nums + log2(min_scale))

#         quats = norm_quats(quats)
#         quats_grid_nums = 2**8
#         quats_clamped = quats.clip(-1, 1)
#         quats_indices = torch.round((quats_clamped + 1) * quats_grid_nums / 2)
#         quantized_quats = (quats_indices / quats_grid_nums * 2 - 1).clip(-1, 1)
#         return quantized_xyz, quantized_opacity, quantized_feature, quantized_scale, quantized_quats
        
#     @property
#     def size(self):
#         if self.valid_size == 0:
#             return 0
#         else:
#             return self.valid_size + 1

#     @torch.no_grad()
#     def add(self, *gaussian):
#         xyzs, opacities, features, scales, quats = unpack_gaussian_parameters(*gaussian)
#         xyzs, opacities, features, scales, quats = self.quantize(xyzs, opacities, features, scales, quats)

#         self.xyz[self.valid_size*2:self.valid_size*2+2] = xyzs
#         self.opacity[self.valid_size*2:self.valid_size*2+2] = opacities
#         self.features[self.valid_size*2:self.valid_size*2+2] = features
#         self.scales[self.valid_size*2:self.valid_size*2+2] = scales
#         self.quats[self.valid_size*2:self.valid_size*2+2] = quats

#         if self.valid_size > 0:
#             sigma, inv_sigma = build_sigma(scales, quats)
#             xyz3, opacity3, feature3, sigma3, inv_sigma3 = merge_gaussian_inv(
#                 xyzs[0], opacities[0], features[0], sigma[0], inv_sigma[0],
#                 xyzs[1], opacities[1], features[1], sigma[1], inv_sigma[1]
#             )
#             scale3, quat3 = unpack_sigma(sigma3)

#             new_features = torch.cat([xyz3, opacity3, feature3, scale3, quat3], dim=-1)
            
#             distances = torch.norm(self.xyz[self.valid_mask] - xyz3[None], dim=-1)
#             index = self.index[self.valid_mask][torch.argmin(distances)]
#             # while item := heapq.heappop(self.pq):
#             #     size, index = item
#             #     if self.valid_mask[index]:
#             #         self.valid_mask[index] = False
#             #         break

#         # size0 = -scales[0, 0] * scales[0, 1] * scales[0, 2]
#         # size1 = -scales[1, 0] * scales[1, 1] * scales[1, 2]

#         # heapq.heappush(self.pq, (size0, self.valid_size*2))
#         # heapq.heappush(self.pq, (size1, self.valid_size*2+1))

#         self.valid_mask[self.valid_size*2:self.valid_size*2+2] = True

#         self.valid_size += 1

#     @torch.no_grad()
#     def get(self):
#         return self.xyz[self.valid_mask], self.opacity[self.valid_mask], self.features[self.valid_mask], self.scales[self.valid_mask], self.quats[self.valid_mask]
    
#     def save_ply(self, path: str) -> None:
#         save_ply(path, *activated_gs2gs(*self.get()))
    
