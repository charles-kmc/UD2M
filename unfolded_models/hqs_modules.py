import torch
from typing import Union
import wandb
import math

from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep, extract_tensor
import utils as utils
import physics as phy

# Half-Quadratic Splitting module       
class HQS_models:
    def __init__(
        self,
        model, 
        physic:Union[phy.Deblurring, phy.Inpainting, phy.SuperResolution], 
        diffusion_scheduler:DiffusionScheduler, 
        get_denoising_timestep:GetDenoisingTimestep,
        args
    ) -> None:
        
        self.model = model
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
        self.get_denoising_timestep = get_denoising_timestep
        self.args = args
    
    def proxf_update(
        self, 
        x:torch.Tensor, 
        t:int,
    )-> torch.Tensor:
        obs = self.obs
        params = self.params
        if self.args.task == "sr":
            zest = self.physic.mean_estimate(obs, x, params)
            out = {"zest":zest}
        elif self.args.task=="deblur":
            zest = self.physic.mean_estimate(obs, x, params)
            out = {"zest":zest}
        elif self.args.task=="inp":
            zest = self.physic.mean_estimate(obs, x, params)
            out = {"zest":zest}
        else:
            raise ValueError("Task not implemented!!")
        
        return out
    
    def run_unfolded_loop(
        self, 
        obs:dict,
        x:torch.Tensor, 
        t:int, 
        max_iter:int = 3,
        T_AUG:int = 5, #15
        lambda_sig:float = 1.0
    ) -> dict:
        self.obs = obs
        x_t, y = self.obs["x_t"], self.obs["y"]
        B,C = x.shape[:2]
        
        # parameters
        sigma_t = extract_tensor(self.diffusion_scheduler.sigmas, t, x_t.shape).ravel()[0]
        rho, t_denoising, ty = self.get_denoising_timestep.get_z_timesteps(t, self.physic.sigma_model, x_t.shape, T_AUG, task = self.args.task)
        sqrt_alpha_comprod = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape).ravel()[0]    
        self.params = {
            "sigma_model":self.physic.sigma_model, 
            "sigma_t":sigma_t, 
            "sqrt_alpha_comprod":sqrt_alpha_comprod,
            "rho":rho,
            "lambda_sig":lambda_sig,
            "ds_max_iter":5,
        }
        
        # pre-compte
        if self.args.task=="sr":
            self.pre_compute = self.physic.pre_calculate(y, self.physic.blur)
          
        # --- initialisation
        x = x / extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t_denoising, x_t.shape).ravel()[0]
        
        for _ in range(max_iter):
            # prox step
            out = self.proxf_update(x, t)
            zest = out["zest"]
            
            # noise prediction
            if self.args.pertub:
                noise = torch.randn_like(zest)
                val = 1 * torch.ones_like(t_denoising).long()
                max_val = 1000 * torch.ones_like(t_denoising).long()
                tu = torch.clamp(t_denoising+0, val, max_val) #3
                z_input = self.diffusion_scheduler.sample_xt(zest, tu, noise=noise)
                wandb.log({"t_denoiser":t_denoising.ravel()[0]})
            else:
                z_input = zest
            
            # Denoising   
            eps = self.model(z_input, t_denoising)
            eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_t.shape[2:]) else (eps, eps)
            
            # estimate of x_start
            x = self.diffusion_scheduler.predict_xstart_from_eps(z_input, t_denoising, eps)
            x_0 = utils.inverse_image_transform(x)
            x_0 = torch.clamp(x_0, 0.0, 1.0)
            
        pred_eps = self.diffusion_scheduler.predict_eps_from_xstrat(x_t, t, x)
        
        wandb.log({"max x":x.max(),"min x":x.min()})
        wandb.log({"max eps":eps.max(),"min eps":eps.min()})
        return {
            "xstart_pred":x_0, 
            "aux":utils.inverse_image_transform(zest), 
            "z_input":z_input, 
            "eps_pred":pred_eps,
            "eps_z":eps
        }
