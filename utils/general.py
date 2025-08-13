import torch
import numpy as np

def log(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    elif isinstance(x, np.ndarray):
        return np.log(x)

def exp(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x)
    elif isinstance(x, np.ndarray):
        return np.exp(x)

def sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        return 1/(1+np.exp(-x))

def inv_sigmoid(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x/(1-x))
    elif isinstance(x, np.ndarray):
        return np.log(x/(1-x))
    
def softplus(x, beta=1.0, threshold=20.0):
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.softplus(x, beta=beta, threshold=threshold)
    elif isinstance(x, np.ndarray):
        return np.log(1+np.exp(beta*x))/beta

def inv_softplus(x, beta=1.0, threshold=20.0):
    if isinstance(x, torch.Tensor):
        return torch.log(torch.exp(beta*x)-1)/beta
    elif isinstance(x, np.ndarray):
        return np.log(np.exp(beta*x)-1)/beta

def normalize_quat(x):
    if isinstance(x, torch.Tensor):
        return x / torch.norm(x, dim=-1, keepdim=True)
    elif isinstance(x, np.ndarray):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

def top_p_sampling(logits, p):
    probs = torch.softmax(logits, dim=-1)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
