import torch
import numpy as np


def merge_gaussian(u1, o1, f1, s1, inv_s1, u2, o2, f2, s2, inv_s2, delta=0.2):
    o3 = o1+o2-o1*o2
    u3 = (o1*u1+o2*u2)/(o1+o2)
    f3 = (o1*f1+o2*f2)/(o1+o2)
    s3 = (o1*s1+o2*s2)/(o1+o2)/np.exp(-delta*np.linalg.norm(u1-u2))*2
    inv_s3 = np.linalg.inv(s3)
    return u3, o3, f3, s3, inv_s3

def merge_gaussian_inv(u1, o1, f1, s1, inv_s1, u2, o2, f2, s2, inv_s2, delta=0.2, multi=1.6):
    if isinstance(u1, torch.Tensor):
        o3 = o1+o2-o1*o2
        u3 = (o1*u1+o2*u2)/(o1+o2)
        f3 = (o1*f1+o2*f2)/(o1+o2)
        inv_s3 = (o1*inv_s1+o2*inv_s2)/(o1+o2)/multi*torch.exp(-delta*torch.norm(u1-u2))
        s3 = torch.linalg.inv(inv_s3)
    elif isinstance(u1, np.ndarray):
        o3 = o1+o2-o1*o2
        u3 = (o1*u1+o2*u2)/(o1+o2)
        f3 = (o1*f1+o2*f2)/(o1+o2)
        inv_s3 = (o1*inv_s1+o2*inv_s2)/(o1+o2)/multi*np.exp(-delta*np.linalg.norm(u1-u2))
        s3 = np.linalg.inv(inv_s3)
    return u3, o3, f3, s3, inv_s3

# def _compute_Gaussian_m0(o, sigma):
#     if isinstance(sigma, torch.Tensor):
#         return o* torch.pi**1.5 * torch.sqrt(torch.linalg.det(sigma))
#     else:
#         return o* np.pi**1.5 * np.sqrt(np.linalg.det(sigma))

def _compute_Gaussian_m0(o, sigma):
    if isinstance(sigma, torch.Tensor):
        log_det = torch.linalg.slogdet(sigma)[1]  # log|sigma|
        log_m0 = torch.log(o) + 1.5 * np.log(2*torch.pi) + 0.5 * log_det
        return torch.exp(log_m0)
    else:
        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            log_det = -np.inf  # 或抛出警告
        log_m0 = np.log(o) + 1.5 * np.log(2*np.pi) + 0.5 * log_det
        return np.exp(log_m0)

def _compute_Gaussian_m1(m0, u):
    return m0 * u

# unused
def _compute_Gaussian_m2_in_x(x, m0, u, s):
    return m0 * ((u-x)[..., None] @ (u-x)[None, ...] + s)

def _compute_Gaussian_cross_(u1, u2, s1, s2):
    if isinstance(u1, torch.Tensor):
        # solve: (inv(s1) + inv(s2)) @ sc = I
        precision = torch.linalg.inv(s1) + torch.linalg.inv(s2)
        try:
            sc = torch.linalg.solve(torch.eye(s1.shape[0], device=s1.device), precision)
        except:
            eps = 1e-6 * torch.eye(s1.shape[0], device=s1.device)
            sc = torch.linalg.solve(torch.eye(s1.shape[0], device=s1.device), precision + eps)
        # uc = sc @ (inv(s1)@u1 + inv(s2)@u2)
        info_vec = torch.linalg.solve(s1, u1) + torch.linalg.solve(s2, u2)
        uc = torch.linalg.solve(precision, info_vec)  # solve(sc^{-1}, info_vec)

    else:  # numpy
        precision = np.linalg.inv(s1) + np.linalg.inv(s2)
        try:
            sc = np.linalg.solve(precision, np.eye(s1.shape[0]))
        except np.linalg.LinAlgError:
            eps = 1e-6 * np.eye(s1.shape[0])
            sc = np.linalg.solve(precision + eps, np.eye(s1.shape[0]))

        info_vec = np.linalg.solve(s1, u1) + np.linalg.solve(s2, u2)
        uc = np.linalg.solve(precision, info_vec)

    return uc, sc

def merge_gaussian_moments(u1, o1, f1, s1, inv_s1, u2, o2, f2, s2, inv_s2, alpha=1.0, cross=False, delta=0.1):
    oc = o1*o2
    uc, sc = _compute_Gaussian_cross_(u1, u2, s1, s2)

    # this is the best way to compute o3
    # ==========================================
    o3 = o1 + o2 - o1*o2
    # ==========================================

    m0_1 = _compute_Gaussian_m0(o1, s1)
    m0_2 = _compute_Gaussian_m0(o2, s2)
    m0 = m0_1 + m0_2
    if cross:
        m0_c = _compute_Gaussian_m0(oc, sc)
        m0 -= m0_c
    
    m1 = _compute_Gaussian_m1(m0_1, u1) + _compute_Gaussian_m1(m0_2, u2)
    if cross:
        m1_c = _compute_Gaussian_m1(m0_c, uc)
        m1 -= m1_c
    
    u3 = m1 / m0

    if isinstance(u1, torch.Tensor):
        inv_s3 = (torch.linalg.inv(s1) * o1 + torch.linalg.inv(s2) * o2) / (o1+o2) / 2 / alpha * torch.exp(-delta*torch.norm(u1-u2)) 
        try:
            s3 = torch.linalg.solve(inv_s3, torch.eye(s1.shape[0], device=s1.device))
        except torch.linalg.LinAlgError:
            eps = 1e-6 * torch.eye(s1.shape[0], device=s1.device)
            s3 = torch.linalg.solve(inv_s3 + eps, torch.eye(s1.shape[0], device=s1.device))

        s3 = (s3 + s3.T) / 2
        s3 = s3 if torch.det(s3) > 0 else -s3
    elif isinstance(u1, np.ndarray):
        # inv_s3 = (np.linalg.inv(s1) * o1 + np.linalg.inv(s2) * o2) / (o1+o2) / 2 / alpha * np.exp(-delta*np.linalg.norm(u1-u2)) 
        inv_s3 = (np.linalg.inv(s1) * o1 + np.linalg.inv(s2) * o2) / (o1+o2) / 2
        try:
            s3 = np.linalg.solve(inv_s3, np.eye(s1.shape[0]))
        except np.linalg.LinAlgError:
            eps = 1e-6 * np.eye(s1.shape[0])
            s3 = np.linalg.solve(inv_s3 + eps, np.eye(s1.shape[0]))
        
        s3 = (s3 + s3.T) / 2
        s3 = s3 if np.linalg.det(s3) > 0 else -s3

    # weighted features
    f3 = (m0_1*f1 + m0_2*f2)/(m0_1+m0_2)
    
    return u3, o3, f3, s3, inv_s3

def merge_gaussian_moments_ub(u1, o1, f1, s1, inv_s1, u2, o2, f2, s2, inv_s2, alpha=1.0, cross=False, delta=0.1):
    oc = o1*o2
    uc, sc = _compute_Gaussian_cross_(u1, u2, s1, s2)

    # this is the best way to compute o3
    # ==========================================
    o3 = o1 + o2 - o1*o2
    # ==========================================

    m0_1 = _compute_Gaussian_m0(o1, s1)
    m0_2 = _compute_Gaussian_m0(o2, s2)
    m0 = m0_1 + m0_2
    if cross:
        m0_c = _compute_Gaussian_m0(oc, sc)
        m0 -= m0_c
    
    m1 = _compute_Gaussian_m1(m0_1, u1) + _compute_Gaussian_m1(m0_2, u2)
    if cross:
        m1_c = _compute_Gaussian_m1(m0_c, uc)
        m1 -= m1_c
    
    u3 = m1 / m0

    if isinstance(u1, torch.Tensor):
        inv_s3 = (torch.linalg.inv(s1) * o1 + torch.linalg.inv(s2) * o2) / (o1+o2) / 2 / alpha * torch.exp(-delta*torch.norm(u1-u2)) 
        try:
            s3 = torch.linalg.solve(inv_s3, torch.eye(s1.shape[0], device=s1.device))
        except torch.linalg.LinAlgError:
            eps = 1e-6 * torch.eye(s1.shape[0], device=s1.device)
            s3 = torch.linalg.solve(inv_s3 + eps, torch.eye(s1.shape[0], device=s1.device))

        s3 = (s3 + s3.T) / 2
        s3 = s3 if torch.det(s3) > 0 else -s3
    elif isinstance(u1, np.ndarray):
        # inv_s3 = (np.linalg.inv(s1) * o1 + np.linalg.inv(s2) * o2) / (o1+o2) / 2 / alpha * np.exp(-delta*np.linalg.norm(u1-u2)) 
        s3 = (m0_1*s1+m0_2*s2)/m0
        inv_s3 = None
        
    # weighted features
    f3 = (m0_1*f1 + m0_2*f2)/(m0_1+m0_2)
    
    return u3, o3, f3, s3, inv_s3

