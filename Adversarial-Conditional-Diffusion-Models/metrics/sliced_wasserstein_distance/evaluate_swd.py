from .swd import swd 
import torch

def eval_swd(samples_ref : torch.tensor, samples_gen : torch.tensor): 
    # (samples_size, width, height) -> (samples_size, 3, width, height
    samples_gen = samples_gen.unsqueeze(1).repeat(1, 3, 1, 1) if len(samples_gen.shape) == 3 else samples_gen
    samples_ref = samples_ref.unsqueeze(1).repeat(1, 3, 1, 1) if len(samples_ref.shape) == 3 else samples_ref
    
    # compute sliced wasserstein distance
    swd_val = swd(samples_ref.to(torch.float64), samples_gen.to(torch.float64))
    return swd_val