import torch 
import deepinv as dinv
import numpy as np
from .motionblur.motionblur import Kernel
        
class ckpt_denoiser(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser
    
    def forward(self, x, sigma):
        if x.requires_grad or sigma.requires_grad:
            def f(x, sigma):
                return self.denoiser(x, sigma)
            out = torch.utils.checkpoint.checkpoint(
                f,
                x,
                sigma,
                use_reentrant=False,
                preserve_rng_state=True
            )
        else: 
            out = self.denoiser(x, sigma)
        return out

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


class random_rot90(torch.nn.Module):
    def forward(self, x):
        rots = int(torch.randint(0,4,[1])[0])
        return torch.rot90(x, k=rots, dims = [-2,-1])



class DiffusionNoise(torch.nn.Module):
    """
        Uniformly samples t~\mathcal{U}([[0, T]]) and adds Gaussian noise to input
        x_0 given the rule
        x_t = x_0 + \sigma_t \mathcal{N}(0,I)
    """
    def __init__(self, T = 1000, beta_start = 0.1/1000, beta_end = 20/1000, device="cpu"):
        super().__init__()
        self.T = T 
        self.sigma = None 
        self.betas = np.linspace(beta_start, beta_end, self.T, dtype=np.float32)
        self.betas = torch.from_numpy(self.betas)
        self.alphas                  = 1.0 - self.betas
        self.alphas_cumprod          = np.cumprod(self.alphas.cpu(), axis=0)
        self.sqrt_alphas_cumprod     = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_1m_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod).to(device)
        self.sigmas   = torch.div(self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod)

    def forward(self, x0, t = None, **kwargs):
        x0 = 2. * x0 - 1
        if t is None:
            self.ts = torch.randint(
                1,
                self.T,
                (x0.shape[0], 1) + (1,) * (x0.dim() - 2),
                device=x0.device
            )
        else: 
            self.ts = torch.tensor(t)
        xt = self.sqrt_alphas_cumprod[self.ts] * x0 + self.sqrt_1m_alphas_cumprod[self.ts] * torch.randn_like(x0)
        # Return output in [0,1]
        self.sigma = 0.5*self.sigmas[self.ts]
        return 0.5*(xt + 1.)

class DiffJointPhysics(dinv.physics.DecomposablePhysics):
    def __init__(self, y_physics:dinv.physics.Physics, diff_physics:DiffusionNoise):
        super().__init__()
        self.y_physics = y_physics
        self.diff_physics = diff_physics
    
    def forward(self, x, t=None, **kwargs):
        y = self.y_physics(x, **kwargs)
        self.sigma_y = self.y_physics.noise_model.sigma
        xt = self.diff_physics(x, t=t)
        self.sigma_xt = self.diff_physics.sigma
        return torch.stack([y, xt], dim=0)

    def prox_l2(self, z, y_xt, gamma):
        r"""
        Implements prox operator in [0,1] domain
        """
        y, xt = y_xt
        xt = xt  # Normalise x_t by \bar\alpha_t^0.5
        rt_alpha_bar_t =  self.diff_physics.sqrt_alphas_cumprod[self.diff_physics.ts]
        alpha_bar_t = rt_alpha_bar_t.square()
        
        b = self.y_physics.A_adjoint(y) / self.sigma_y**2 \
            + 1 / gamma * z \
            + 4 * rt_alpha_bar_t*xt/(1-alpha_bar_t)\
            - 2 *rt_alpha_bar_t*(1-rt_alpha_bar_t)/(1-alpha_bar_t)  # Bias from moving xt back to [0,1]

        if isinstance(self.y_physics.mask, float):
            scaling = self.y_physics.mask**2/ self.sigma_y**2 + 1 / gamma  + 1 / self.sigma_xt**2 * xt
        else:  # Dinv uses rfft for singular value decomposition of circular blur
            scaling = torch.conj(self.y_physics.mask) * self.y_physics.mask/ self.sigma_y**2 
            scaling = scaling.expand(self.sigma_xt.shape[0],*scaling.shape[1:])
            scaling = scaling + 1 / self.sigma_xt.clone().unsqueeze(-1)**2 + 1 / gamma
        # Solve {scaling * x = b}
        x = self.y_physics.V(self.y_physics.V_adjoint(b) / scaling)
        return x