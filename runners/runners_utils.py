import torch 
import numpy as np

import unfolded_models as ums
import utils as utils


class DiffusionSolver:
    def __init__(self, seq, diffusion:ums.DiffusionScheduler):
        self.seq = seq
        self.diffusion = diffusion
    
    # DDIM solver   
    def ddim_solver(self, x0, eps, ii, eta, zeta):
        t_i = self.seq[ii]
        t_im1 = self.seq[ii+1]
        beta_t = self.diffusion.betas[self.seq[ii]]
        eta_sigma = eta * self.diffusion.sqrt_1m_alphas_cumprod[t_im1] / self.diffusion.sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(beta_t)
        x = self.diffusion.sqrt_alphas_cumprod[t_im1] * x0 \
            + (torch.sqrt(self.diffusion.sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps + eta_sigma * torch.randn_like(x0)) \
                + eta_sigma * self.diffusion.sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x0)
        return x
    
    # DDPM solver
    def ddpm_solver(self, x, eps, ii, ddpm_param = 1):
        t_i = self.seq[ii]
        t_im1 = self.seq[ii+1]
        eta_sigma = self.diffusion.sqrt_1m_alphas_cumprod[t_im1] / self.diffusion.sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(self.diffusion.betas[t_i])
        alpha_t = 1 - self.diffusion.betas[t_i]
        x = (1 / alpha_t.sqrt()) * ( x - (ddpm_param*self.diffusion.betas[t_i]).sqrt() * eps ) + (ddpm_param*eta_sigma**2 ).sqrt() * torch.randn_like(x)
        
        return x
    
    def huen_solver0(self, x0, x_t, ii, eps_old=None, scaling=None):
        t = self.seq[ii]
        eps_new = self.diffusion.predict_eps_from_xstrat(x_t, t, x0)
        if eps_old is not None:
            eps_new = 0.5*(eps_new + eps_old) 
        if eps_old is not None:
            beta_t = self.diffusion.betas[self.seq[ii-1]]
        else:
            beta_t = self.diffusion.betas[self.seq[ii]]
        x_tm1 = ( x_t - beta_t.sqrt() * eps_new ) / (1-beta_t).sqrt()
        if eps_old is None:
            return x_tm1, eps_new
        else: 
            return x_tm1
        
    # TODO replace ii with tii 
    def huen_solver(self, x_t, d_new, ii, d_old=None):    
        t_ii = self.seq[ii - 1] if d_old is not None else self.seq[ii]  
        if d_old is not None:
            d_new = (d_new + d_old) / 2  
        x_tiim1 = (x_t - beta_t.sqrt() * d_new) / (1 - beta_t).sqrt()
        beta_t = ums.extract_tensor(self.diffusion.betas, t_ii, x_t.shape)
        return (x_tiim1, d_new) if d_old is None else x_tiim1