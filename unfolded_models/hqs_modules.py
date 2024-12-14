import torch
from typing import Union
import wandb

from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep, extract_tensor
import utils as utils
import physics as phy

# Half-Quadratic Splitting module       
class HQS_models:
    def __init__(
        self,
        model, 
        physic:Union[phy.Deblurring, phy.Inpainting], 
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
        y:torch.Tensor,
        x:torch.Tensor, 
        x_t:torch.Tensor, 
        rho:float, 
        t:int,
        lambda_:float,
    )-> torch.Tensor:
        sigma_t = extract_tensor(self.diffusion_scheduler.sigmas, t, x_t.shape)
        sqrt_alpha_comprod = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape)
        estimate_temp = self.physic.mean_estimate(x_t, y, x, self.physic.sigma_model, sigma_t, sqrt_alpha_comprod, rho,lambda_)
        return estimate_temp
    
    def run_unfolded_loop(
        self, 
        y:torch.Tensor, 
        x:torch.Tensor, 
        x_t:torch.Tensor, 
        t:int, 
        max_iter:int = 3,
        T_AUG:int = 0,
        lambda_:float = 1.0
    ) -> dict:
        B,C = x.shape[:2]
        
        # get denoising timestep
        rho, t_denoising, ty = self.get_denoising_timestep.get_tz_rho(t, self.physic.sigma_model, x_t.shape, T_AUG, task = self.args.task)
        
        # --- initialisation
        x = x / extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t_denoising, x.shape)
            
        if self.args.use_wandb and self.args.mode == "train":
            wandb.log(
                {
                    "tz":t_denoising.squeeze().max().cpu().numpy(), 
                    "rho":rho.squeeze().min().cpu().numpy(),
                }
            )
            
        for _ in range(max_iter):
            # prox step
            z_ = self.proxf_update(y, x, x_t, rho, t, lambda_)

            # noise prediction
            if self.args.pertub:
                noise = torch.randn_like(z_)
                val = 1 * torch.ones_like(t_denoising).long()
                max_val = 1000 * torch.ones_like(t_denoising).long()
                tu = torch.clamp(t_denoising-3, val, max_val)
                z = self.diffusion_scheduler.sample_xt(z_, tu, noise=noise)
            else:
                z = z_.clone()
                
            eps = self.model(z, t_denoising)
            eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_t.shape[2:]) else (eps, eps)
            
            # estimate of x_start
            x = self.diffusion_scheduler.predict_xstart_from_eps(z, t_denoising, eps)
            x_pred = utils.inverse_image_transform(x)
            x_ = torch.clamp(x_pred, 0.0, 1.0)
            
        pred_eps = self.diffusion_scheduler.predict_eps_from_xstrat(x_t, t, x)
        return {
            "xstart_pred":x_, 
            "aux":utils.inverse_image_transform(z_), 
            "eps_pred":pred_eps,
            "eps_z":eps
        }
