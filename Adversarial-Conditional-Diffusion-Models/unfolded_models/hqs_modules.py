import torch
from typing import Union
import wandb

from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep
import utils as utils
import physics as phy
import models as models



T_AUG = 10

# Half-Quadratic Splitting module       
class HQS_models(torch.nn.Module):
    def __init__(
        self,
        model, 
        physic:Union[phy.Deblurring, phy.Inpainting, phy.SuperResolution], 
        diffusion_scheduler:DiffusionScheduler, 
        get_denoising_timestep:GetDenoisingTimestep,
        args,
        device,
        max_iter:int = 3,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_iter = max_iter
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
        self.get_denoising_timestep = get_denoising_timestep
        self.args = args
        self.SP_Param = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(max_iter)])
        self.LambdaSIG= torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(max_iter)])
        self.TAUG = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(max_iter)])  # Used to augment the denoising step
        self.TLAT = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(max_iter)])  # Used to auhment noise added before the denosiing step
    
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
        elif self.args.task=="ct":
            zest = self.physic.mean_estimate(obs, x, x, params)
            out = {"zest":zest}
        else:
            raise ValueError("Task not implemented!!")
        
        return out

    
    
    def run_unfolded_loop(
        self, 
        obs:dict,
        x:torch.Tensor, 
        t:int, 
        T_AUG:int = T_AUG, #15
        lambda_sig:float = 1.0,
        checkpoint = True,
        **kwargs
    ) -> dict:
        self.obs = obs
        x_t, y = self.obs["x_t"], self.obs["y"]
        B,C = x.shape[:2]

        y_label = self.obs["y_label"] if "y_label" in self.obs.keys() else None
    
        
        
        for ii in range(self.max_iter):
            # parameters
            sigma_t = models.extract_tensor(self.diffusion_scheduler.sigmas, t, x_t.shape).ravel()[0]
            lambda_sig = self.LambdaSIG[ii].exp()
            rho, t_denoising, ty = self.get_denoising_timestep.get_z_timesteps(t, self.physic.sigma_model, x_t.shape, T_AUG = 1000*self.TAUG[ii], sp = self.SP_Param[ii].exp())
            sqrt_alpha_comprod = models.extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape).ravel()[0]    
            self.params = {
                "sigma_model":self.physic.sigma_model, 
                "sigma_t":sigma_t, 
                "sqrt_alpha_comprod":sqrt_alpha_comprod,
                "rho":rho,
                "lambda_sig":lambda_sig,
                "ds_max_iter":5,
            }
            if self.args.task=="ct":
                self.params["max_iter_algo"] = self.args.physic.ct.max_iter_algo
                self.params["step_algo"] =  self.args.physic.ct.step_algo
        
            # --- initialisation
            # x = x / models.extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t_denoising, x_t.shape).ravel()[0]
            # prox step
            if ii==self.max_iter-1 and self.args.mode!="train":
                self.params["lambda_sig"]=lambda_sig
            elif ii==self.max_iter-2 and self.args.mode!="train":
                self.params["lambda_sig"]=lambda_sig
            else:
                pass
            out = self.proxf_update(x, t)
            zest = out["zest"]
            
            # noise prediction
            if self.args.pertub:
                noise = torch.randn_like(zest)
                val = 1 * torch.ones_like(t_denoising).long()
                max_val = 1000 * torch.ones_like(t_denoising).long()
                tu = torch.clamp(t_denoising + 1000*self.TLAT[ii], val, max_val) 
                z_input = self.diffusion_scheduler.sample_xt_cont(zest, tu, noise=noise)
                # if self.args.use_wandb:
                #     wandb.log({"t_denoiser":t_denoising.ravel()[0]})
            else:
                z_input = zest
            
            # Denoising   
            if checkpoint:
                def run_model(z, t):
                    return self.model(z, t, y = y_label)
                eps = torch.utils.checkpoint.checkpoint(run_model, z_input, t_denoising, use_reentrant=False)
            else:
                eps = self.model(z_input, t_denoising, y=y_label)
            eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_t.shape[2:]) else (eps, eps)
            
            # estimate of x_start
            x = self.diffusion_scheduler.predict_xstart_from_eps_cont(z_input, t_denoising, eps)
            
            x_0 = x.add(1).div(2)
            x_0 = torch.clamp(x_0, 0.0, 1.0)
            
            if self.args.task=="inp" and ii<self.max_iter-1:
                x = (1-self.physic.Mask)*x + (self.physic.Mask)*y
        
        pred_eps = self.diffusion_scheduler.predict_eps_from_xstrat(x_t, t, x)
        if self.args.use_wandb:
            wandb.log({"max x":x.max(),"min x":x.min()})
            wandb.log({"max eps":eps.max(),"min eps":eps.min()})
        return {"xstart_pred":x_0, "aux":torch.clamp(zest.add(1).div(2),0,1), "z_input":z_input, "eps_pred":pred_eps,"eps_z":eps}
