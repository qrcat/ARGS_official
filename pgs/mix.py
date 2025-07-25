import numpy as np


def mix_gaussian(u1, o1, f1, s1, inv_s1, u2, o2, f2, s2, inv_s2, delta=0.2):
    o3 = o1+o2-o1*o2
    u3 = (o1*u1+o2*u2)/(o1+o2)
    f3 = (o1*f1+o2*f2)/(o1+o2)
    s3 = (o1*s1+o2*s2)/(o1+o2)/np.exp(-delta*np.linalg.norm(u1-u2))*2
    inv_s3 = np.linalg.inv(s3)
    return u3, o3, f3, s3, inv_s3

def mix_gaussian_inv(u1, o1, f1, s1, inv_s1, u2, o2, f2, s2, inv_s2, delta=0.2):
    o3 = o1+o2-o1*o2
    u3 = (o1*u1+o2*u2)/(o1+o2)
    f3 = (o1*f1+o2*f2)/(o1+o2)
    inv_s3 = (o1*inv_s1+o2*inv_s2)/(o1+o2)/2*np.exp(-delta*np.linalg.norm(u1-u2))
    s3 = np.linalg.inv(inv_s3)
    return u3, o3, f3, s3, inv_s3

