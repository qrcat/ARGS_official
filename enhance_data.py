
from utils.io import activated_gs2gs, gs2activated_gs, save_ply, load_ply
from utils.shs import RGB2SH, SH2RGB
from utils.gaussian import norm_quats

from torchvision.transforms import ColorJitter
from pytorch3d.transforms import quaternion_multiply, matrix_to_quaternion, quaternion_to_matrix

import numpy as np
import torch


class Augment:
    color_stage = {
        "low": { "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.05,
        },
        "middle": {
            "brightness": 0.2,
            "contrast": 0.3,
            "saturation": 0.4,
            "hue": 0.1,
        },
        "high": {
            "brightness": 0.4,
            "contrast": 0.5,
            "saturation": 0.6,
            "hue": 0.2,
        }
    }

    @classmethod
    def jitter_point_cloud(cls, xyz, sigma=0.01, prob=0.95):
        """
        Jitter the point clouds\n
        Input:
            Nx3 tensor, original point clouds\n
        Output:
            Nx3 tensor, jittered point clouds
        """
        if torch.rand([])<prob:
            xyz += torch.randn_like(xyz) * sigma

        return xyz
    
    @classmethod
    def jitter_opacities(cls, opacities, sigma=0.01, prob=0.95):
        """
        Jitter the opacities of the point clouds\n
        Input:
            Nx1 tensor, original opacities\n
        Output:
            Nx1 tensor, jittered opacities
        """
        if torch.rand([])<prob:
            opacities += torch.randn_like(opacities) * sigma

        return opacities
    
    @classmethod
    def jitter_color(cls, shs, brightness=0.1, contrast=0.1, saturation=0.2, hue=0.05, prob=0.95):
        """
        Jitter the colors of the point clouds\n
        Input:
            Nx3 tensor, original colors\n
        Output:
            Nx3 tensor, jittered colors
        """
        if torch.rand([])<prob:
            jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        
            rgb = SH2RGB(shs) # convert back to rgb
            rgb = rgb[None].permute(2, 0, 1)
            rgb = jitter(rgb)
            rgb = rgb.permute(1, 2, 0)
            shs = RGB2SH(rgb[0]) # convert back to sh

        return shs

    @classmethod
    def jitter_scale(cls, scales, sigma=0.01, prob=0.95):
        """
        Jitter the scales of the point clouds\n
        Input:
            Nx3 tensor, original scales\n
        Output:
            Nx3 tensor, jittered scales
        """
        if torch.rand([])<prob:
            scales += torch.randn_like(scales) * sigma

        return scales

    @classmethod
    def rotate_point_cloud(cls, xyz, rots, max_angle=2*np.pi, upaxis=3, prob=0.95):
        """
        Rotate the point clouds\n
        Input:
            [Nx3 tensor, Nx4 tensor], original point clouds and rotations\n
        Output:
            [Nx3 tensor, Nx4 tensor], rotated point clouds and rotations
        """
        if torch.rand([])<prob:
            # angle = torch.rand([1])*max_angle
            angle = torch.tensor([np.pi/3])
            if upaxis==1:
                R = cls.rot_x(angle)
            elif upaxis==2:
                R = cls.rot_y(angle)
            elif upaxis==3:
                R = cls.rot_z(angle)
            else:
                raise ValueError('unkown spatial dimension')
            # apply rotation
            xyz = torch.matmul(xyz, R.T)
            # update rotation
            q = matrix_to_quaternion(R)
            rots = quaternion_multiply(q, rots)

        return xyz, rots

    @classmethod
    def rotate_perturbation_point_cloud(cls, xyz, rots, angle_sigma=0.06, prob=0.95):
        """
        Rotate the point clouds with perturbation\n
        Input:
            [Nx3 tensor, Nx4 tensor], original point clouds and rotations\n
        Output:
            [Nx3 tensor, Nx4 tensor], rotated point clouds and rotations
        """
        if torch.rand([])<prob:
            angle = torch.randn([1,3])*angle_sigma
            Rx = cls.rot_x(angle[:,0])
            Ry = cls.rot_y(angle[:,1])
            Rz = cls.rot_z(angle[:,2])
            R = torch.matmul(Rz, torch.matmul(Ry, Rx))
            # apply rotation
            xyz = torch.matmul(xyz, R.T)
            # update rotation
            q = matrix_to_quaternion(R)
            rots = quaternion_multiply(q, rots)

        return xyz, rots

    @staticmethod
    def flip_point_cloud(xyz, rots, prob=0.95):
        """
        Flip the point clouds\n
        Input:
            [Nx3 tensor, Nx4 tensor], original point clouds and rotations\n
        Output:
            [Nx3 tensor, Nx4 tensor], flipped point clouds and rotations
        """
        if torch.rand([])<prob:
            x = xyz[:,0]
            y = xyz[:,1]
            z = xyz[:,2]

            rand = torch.rand([])
            # xy, xz, yz

            if rand<1/4:
                x = - x  # flip x-dimension(horizontal)
                rots = torch.stack([-rots[..., 0], -rots[...,1 ], rots[..., 2], rots[..., 3]], dim=-1)
            elif rand < 1/2:
                y = -y
                rots = torch.stack([-rots[..., 0], rots[..., 1], -rots[..., 2], rots[..., 3]], dim=-1)
            else:
                # do nothing 
                pass
            
            xyz = torch.stack([x, y, z], dim=1)
        return xyz, rots

    @staticmethod
    def random_scale_point_cloud(xyz, scales, scale_low=0.8, scale_high=1.2, prob=0.95):
        if torch.rand([])<prob:
            _scales = torch.rand([])*(scale_high-scale_low) + scale_low
            xyz *= _scales
            scales *= _scales
        return xyz, scales

    @staticmethod
    def rot_x(angle):
        cosval = torch.cos(angle)
        sinval = torch.sin(angle)
        val0 = torch.zeros_like(cosval)
        val1 = torch.ones_like(cosval)
        R = torch.stack([val1, val0, val0,
                         val0, cosval, -sinval,
                         val0, sinval, cosval], dim=1)
        R = torch.reshape(R, (3, 3))
        return R

    @staticmethod
    def rot_y(angle):
        cosval = torch.cos(angle)
        sinval = torch.sin(angle)
        val0 = torch.zeros_like(cosval)
        val1 = torch.ones_like(cosval)
        R = torch.stack([cosval, val0, sinval,
                         val0, val1, val0,
                        -sinval, val0, cosval], dim=1)
        R = R.view(3, 3)
        return R

    @staticmethod
    def rot_z(angle):
        cosval = torch.cos(angle)
        sinval = torch.sin(angle)
        val0 = torch.zeros_like(cosval)
        val1 = torch.ones_like(cosval)
        R = torch.stack([cosval, -sinval, val0,
                         sinval, cosval, val0,
                         val0, val0, val1], dim=1)
        R = R.view(3, 3)
        return R
    
    def __call__(
            self, 
            xyz,
            opacities,
            features_dc,
            scales,
            quats,
            prob=0.5,
            upaxis=3,
            div_factor=10,
            color_enhance_mode="middle",
        ):
        # jitter position
        xyz = Augment.jitter_point_cloud(xyz, sigma=torch.mean(scales, dim=-1, keepdim=True)/div_factor, prob=prob)
        # jitter opacities
        opacities = Augment.jitter_opacities(opacities, sigma=0.01, prob=prob)
        # jitter colors
        features_dc = Augment.jitter_color(features_dc, prob=prob, **self.color_stage[color_enhance_mode])
        # # jitter scales
        xyz, scales = Augment.random_scale_point_cloud(xyz, scales, prob=prob)
        scales = Augment.jitter_scale(scales, sigma=scales/div_factor, prob=prob)
        # jitter rotation
        xyz, quats = Augment.rotate_point_cloud(xyz, quats, upaxis=upaxis, prob=prob)
        xyz, quats = Augment.rotate_perturbation_point_cloud(xyz, quats, angle_sigma=0.1, prob=prob)
        xyz, quats = Augment.flip_point_cloud(xyz, quats, prob=prob)
        
        return xyz, opacities, features_dc, scales, quats


def enhance_gaussian_field(path: str, save: str, augment: Augment, prob=0.5, upaxis=3, color_enhance_mode="high"):
    xyz, opacities, features_dc, scales, rots = gs2activated_gs(*load_ply(path))
    # convert to torch tensor
    xyz = torch.from_numpy(xyz).float()
    opacities = torch.from_numpy(opacities).float()
    features_dc = torch.from_numpy(features_dc).float()
    scales = torch.from_numpy(scales).float()
    rots = torch.from_numpy(rots).float()
    # normalize the rotation
    rots = norm_quats(rots)
    # augment the gaussian field
    xyz, opacities, features_dc, scales, rots = augment(
        xyz, opacities, features_dc, scales, rots, 
        upaxis=upaxis, prob=prob, color_enhance_mode=color_enhance_mode,
    )
    # save the enhanced gaussian field
    xyz, opacities, features_dc, scales, rots = activated_gs2gs(xyz, opacities, features_dc, scales, rots)
    save_ply(save, xyz, opacities, features_dc, scales, rots)

augment = Augment()
# for shapesplat and modelsplat, upaxis = 3
enhance_gaussian_field('03636649-cfaf30102d9b7cc6cd6d67789347621.ply', '03636649-enhance.ply', augment)
