import torch
from .quaternion import quaternion_multiply, quaternion_inverse, normalize_quaternions

# ===========================
# Local/Global transforms
# ===========================
def to_local(parent: torch.Tensor, children: torch.Tensor) -> torch.Tensor:
    """
    parent:   (M, 14)
    children: (M, 2, 14)
    return:   (M, 2, 14)  order:
      Δpos(3), Δopacity(ratio)(1), Δfeat(3), Δscale(log ratio)(3), Δquat(relative)(4)
    """
    eps = 1e-6
    local = torch.zeros_like(children)

    # Δpos
    local[..., :3] = children[..., :3] - parent[:, None, :3]
    # Δopacity (ratio)
    p_op = torch.clamp(parent[:, 3:4], min=eps)
    c_op = torch.clamp(children[..., 3:4], min=eps)
    local[..., 3:4] = c_op / p_op[:, None, :]
    # Δfeature (difference)
    local[..., 4:7] = children[..., 4:7] - parent[:, None, 4:7]
    # Δscale = log(child/parent)
    p_s = torch.clamp(parent[:, 7:10], min=eps)
    c_s = torch.clamp(children[..., 7:10], min=eps)
    local[..., 7:10] = torch.log(c_s) - torch.log(p_s[:, None, :])
    # Δquat = q_parent^{-1} ⊗ q_child
    q_parent = normalize_quaternions(parent[:, -4:])
    q_child0 = normalize_quaternions(children[:, 0, -4:])
    q_child1 = normalize_quaternions(children[:, 1, -4:])
    q_local0 = quaternion_multiply(quaternion_inverse(q_parent), q_child0)
    q_local1 = quaternion_multiply(quaternion_inverse(q_parent), q_child1)
    local[:, 0, -4:] = normalize_quaternions(q_local0)
    local[:, 1, -4:] = normalize_quaternions(q_local1)
    return local

def to_global(parent: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """
    parent: (M, 14)
    local:  (M, 2, 14)
    return: (M, 2, 14) global children
    """
    out = torch.zeros_like(local)
    # pos
    out[..., :3] = parent[:, None, :3] + local[..., :3]
    # opacity (ratio * parent)
    out[..., 3:4] = parent[:, None, 3:4] * local[..., 3:4]
    # feature
    out[..., 4:7] = parent[:, None, 4:7] + local[..., 4:7]
    # scale = parent * exp(Δ)
    out[..., 7:10] = torch.exp(local[..., 7:10] + torch.log(parent[:, None, 7:10]))
    # quat = q_parent ⊗ Δq
    q_parent = normalize_quaternions(parent[:, -4:])
    q_loc0 = normalize_quaternions(local[:, 0, -4:])
    q_loc1 = normalize_quaternions(local[:, 1, -4:])
    out[:, 0, -4:] = quaternion_multiply(q_parent, q_loc0)
    out[:, 1, -4:] = quaternion_multiply(q_parent, q_loc1)
    out[:, :, -4:] = normalize_quaternions(out[:, :, -4:])
    return out
