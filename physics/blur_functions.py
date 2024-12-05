import torch
import numpy as np
from deepinv.physics.generator import MotionBlurGenerator
from .motionblur import Kernel

def motion_kernel(kernel_size:int, sigma:float = 0.3, l:float = 0.5)->torch.Tensor:
    generator = MotionBlurGenerator((kernel_size,kernel_size), num_channels=1) 
    kernel = generator.step(sigma, l)
    return kernel["filter"].squeeze()
    
def gaussian_kernel(kernel_size:int, dtype:torch.dtype, std:int = 3.0):
    c = int((kernel_size - 1)/2)
    v = torch.arange(-kernel_size + c, kernel_size - c, dtype = dtype).reshape(-1,1)
    vT = v.T
    kernel_ = torch.exp(-0.5*(v**2 + vT**2)/std)
    kernel_ /= torch.sum(kernel_)
    return kernel_ 
    
def uniform_kernel(kernel_size:int, dtype:torch.dtype):
    kernel_ = torch.ones(kernel_size, kernel_size).to(dtype)
    kernel_ /= torch.sum(kernel_)
    return kernel_

class UniformMotionBlurGenerator(torch.nn.Module):
    def __init__(
            self,
            kernel_size = (61, 61),
            intensity = 0.5,
            seed = 1
        ):
        super().__init__()
        self.kernel_size = kernel_size 
        self.intensity = intensity 
        self.rng = np.random.default_rng(seed)
    
    def step(self, batch_size):
        kernels = torch.stack(
            [
                torch.from_numpy(
                    Kernel(size=self.kernel_size, intensity=self.intensity ,rng=self.rng).kernelMatrix
                ).unsqueeze(0)
                for i in range(batch_size)
            ],
            dim=0
        )
        kernels.requires_grad = False
        return {"filter":kernels}
    
# Memory-efficient Dataset for inverse problems - samples measurements at batch time
# But requires more computation to load batches

def get_motion_blur(kernel_no):
    k = "kernel_" + str(kernel_no)
    import json 
    import numpy as np
    with open("data/Levinkernels.json") as infile:
        motionblur = json.load(infile)
    kernel = torch.from_numpy(np.array(motionblur[k]))
    kernel.requires_grad = False
    return kernel.unsqueeze(0).unsqueeze(0)
