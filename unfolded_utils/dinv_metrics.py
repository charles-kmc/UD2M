import torch
from deepinv.loss.metric import LPIPS, SSIM
import numpy as np
import random
import os

def get_rng_state():
    try:
        pythonhash = os.environ["PYTHONHASHSEED"]
    except:
        pythonhash = KeyError("No python hash seed set")
    return {
        "np_state":np.random.get_state(),
        "random_state":random.getstate(),
        "torch_state":torch.get_rng_state(),
        "cuda_state":torch.cuda.get_rng_state(),
        "cuda_all_state":torch.cuda.get_rng_state_all(),
        "cudnn_benchmark":torch.backends.cudnn.benchmark,
        "cudnn_deterministic":torch.are_deterministic_algorithms_enabled(),
        "pythonhash":"" if isinstance(pythonhash, KeyError) else pythonhash
    }

def set_rng_state(
    np_state=None,
    random_state=None,
    torch_state=None,
    cuda_state=None,
    cuda_all_state=None,
    cudnn_benchmark=None,
    cudnn_deterministic=None,
    pythonhash=None
):
    np.random.set_state(np_state)
    random.setstate(random_state)
    torch.set_rng_state(torch_state)
    torch.cuda.set_rng_state(cuda_state)
    torch.cuda.set_rng_state_all(cuda_all_state)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    os.environ["PYTHONHASHSEED"]=pythonhash


class LPIPS_seed(LPIPS):
    def __init__(self, **kwargs):
        rng_state = get_rng_state()  # Get current rng state
        super().__init__(**kwargs)
        set_rng_state(**rng_state)  # Restore rng state
    
    def forward(self, x, x_net, **kwargs):
        rng_state = get_rng_state() # Get current rng state
        out = super().forward(x, x_net, **kwargs)
        set_rng_state(**rng_state)  # Restore rng state
        return out

class SSIM_seed(SSIM):
    def __init__(self, **kwargs):
        rng_state = get_rng_state()  # Get current rng state
        super().__init__(**kwargs)
        set_rng_state(**rng_state)  # Restore rng state
    
    def forward(self, x, x_net, **kwargs):
        rng_state = get_rng_state() # Get current rng state
        out = super().forward(x, x_net, **kwargs)
        set_rng_state(**rng_state)  # Restore rng state
        return out

    


