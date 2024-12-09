import torch 
import numpy as np
import yaml
import blobfile as bf

import unfolded_models as ums
import utils as utils

class DiffusionSolver:
    def __init__(self, seq, diffusion:ums.DiffusionScheduler, ):
        self.seq = seq
        self.diffusion = diffusion
    
    # DDIM solver   
    def ddim_solver(self, x0, eps, ii, eta, zeta):
        t_i = self.seq[ii]
        t_im1 = self.seq[ii+1]
        beta_t = self.diffusion.betas[self.seq[ii]]
        eta_sigma = eta * self.diffusion.sqrt_1m_alphas_cumprod[t_im1] / self.diffusion.sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(beta_t)
        x = self.diffusion.sqrt_alphas_cumprod[t_im1] * x0 \
            + np.sqrt(1-zeta) * (torch.sqrt(self.diffusion.sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps + eta_sigma * torch.randn_like(x0)) \
                + np.sqrt(zeta) * self.diffusion.sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x0)
        return x
    
    # DDPM solver
    def ddpm_solver(self, x, eps, ii, ddpm_param = 1):
        t_i = self.seq[ii]
        t_im1 = self.seq[ii+1]
        eta_sigma = self.diffusion.sqrt_1m_alphas_cumprod[t_im1] / self.diffusion.sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(self.diffusion.betas[t_i])
        alpha_t = 1 - self.diffusion.betas[t_i]
        x = (1 / alpha_t.sqrt()) * ( x - (ddpm_param*self.diffusion.betas[t_i]).sqrt() * eps ) + (ddpm_param*eta_sigma**2 ).sqrt() * torch.randn_like(x)
        
        return x
        
# load lara weights                
def load_trainable_params(model, epoch, args):
    checkpoint_dir = bf.join(args.path_save, args.task, "Checkpoints", args.date, f"model_noise_{args.physic.sigma_model}")
    filename = f"LoRA_model_{args.task}_{(epoch):03d}.pt"
    filepath = bf.join(checkpoint_dir, filename)
    
    trainable_state_dict = torch.load(filepath)
    model.load_state_dict(trainable_state_dict["model_state_dict"], strict=False)               
    
    return model

# load yaml file 
def load_yaml(save_checkpoint_dir, epoch, date, task):
    dir_ = bf.join(save_checkpoint_dir, date)
    filename = f"LoRA_model_{task}_{(epoch):03d}.yaml"
    filepath = bf.join(dir_, filename)
    with open(filepath, 'r') as f:
        args = yaml.safe_load(f)
    
    return args

