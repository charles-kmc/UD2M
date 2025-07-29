import torch
from typing import Union
import wandb

from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep
import utils as utils
import physics as phy
import models as models
import deepinv as dinv

T_AUG = 10
class DDNoise(dinv.physics.DecomposablePhysics):
    def __init__(self, scheduler,  t = 100):
        super().__init__()
        self.scheduler = scheduler
        self.alphas_cumprod = scheduler.alphas_cumprod
        self.device = self.alphas_cumprod.device
        self.set_t(t)
    
    def set_t(self, t):
        self.mask = torch.ones((1, 3, 256, 256), device=self.device)*(self.alphas_cumprod[t].sqrt())
        self.noise_model = dinv.physics.GaussianNoise((1 - self.alphas_cumprod[t]).sqrt()/2).to(self.device)

    
    def forward(self, x, t = None):
        if t is not None:
            self.set_t(t)
        return super().forward(x, self.mask)

class TrainableGaussianNoise(dinv.physics.GaussianNoise):
    def __init__(self, sigma):
        super().__init__(sigma)
        del self.sigma
        self.sigma = sigma
    
    def forward(self, x, **kwargs):
        return super().forward(x, self.sigma)
    
    def update_parameters(self, sigma=None, **kwargs):
        self.sigma = sigma

class StackedJointPhysics(dinv.physics.StackedLinearPhysics):
    def __init__(self, physics, scheduler, prox = False):
        dd_physics = DDNoise(scheduler)
        physics_list = [physics, dd_physics]
        if prox: 
            physics_list.append(dinv.physics.DecomposablePhysics(noise_model=TrainableGaussianNoise(0.0)))
        super(StackedJointPhysics, self).__init__(physics_list)
        self.noise_model = dinv.physics.GaussianNoise(torch.min(torch.tensor([p.noise_model.sigma for p in physics_list])))
    
    def set_t(self, t):
        self.physics_list[1].set_t(t)
        self.noise_model.update_parameters(sigma=torch.min(torch.tensor([p.noise_model.sigma for p in self.physics_list])))
    
    def set_rho(self, rho):
        self.physics_list[2].noise_model.update_parameters(rho.mean())
        self.noise_model.sigma = torch.nn.Parameter(torch.min(torch.tensor([p.noise_model.sigma for p in self.physics_list])))

    def A(self, x, **kwargs):
        min = torch.min(torch.tensor([p.noise_model.sigma for p in self.physics_list]))
        return dinv.utils.TensorList([
            physics.A(x, **kwargs)*min/physics.noise_model.sigma
            for physics in self.physics_list
        ])

    def A_adjoint(self, y, **kwargs) -> torch.Tensor:
        r"""
        Computes the adjoint of the stacked operator, defined as
        .. math::
            A^{\top}y = \sum_{i=1}^{n} A_i^{\top}y_i.
        :param deepinv.utils.TensorList y: measurements
        """
        min = torch.min(torch.tensor([p.noise_model.sigma for p in self.physics_list]))
        return self.reduction(
            [
                physics.A_adjoint(y[i], **kwargs)*min/physics.noise_model.sigma
                for i, physics in enumerate(self.physics_list)
            ]
        )

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
        max_unfolded_iter:int = 1,
        iter_dep_weights=False,
        RAM = None,
        RAM_Physics = None,
        RAM_prox = False,
        ) -> None: 
        super().__init__()
        self.model = model
        self.max_unfolded_iter = args.max_unfolded_iter
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
        self.get_denoising_timestep = get_denoising_timestep
        self.args = args
        self.num_weights_hqs = self.max_unfolded_iter if iter_dep_weights else 1
        self.SP_Param = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(self.num_weights_hqs)])
        self.LambdaSIG= torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=RAM is None) for _ in range(self.num_weights_hqs)])  # Used to augment the denoising step
        self.TAUG = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(self.num_weights_hqs)])  # Used to augment the denoising step
        self.TLAT = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(self.num_weights_hqs)])  # Used to auhment noise added before the denosiing step
        self.RAM = RAM
        self.RAM_prox = RAM_prox and RAM is not None
        if RAM_Physics is not None:
            self.RAM_physics = StackedJointPhysics(RAM_Physics, self.diffusion_scheduler, prox = True)

    def proxf_update(
        self, 
        x:torch.Tensor, 
        t:int,
    )-> torch.Tensor:
        obs = self.obs
        params = self.params
        if self.RAM_prox:
            def run_prox(x):
                return self.RAM_prox_step(obs["y"], obs["x_t"], t, x)
            x = torch.utils.checkpoint.checkpoint(run_prox, x, use_reentrant=False)
            return {"zest":x}
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


    def RAM_prox_step(self, y, xt, t, z):
        if isinstance(t, torch.Tensor) and len(t)>1:
            t = int(t[0].item())
        self.RAM_physics.set_t(t)
        self.RAM_physics.set_rho(self.params["rho"])
        xt_init = (xt.clone() + self.RAM_physics.physics_list[1].alphas_cumprod[t].sqrt())/2 
        min = torch.min(torch.tensor([p.noise_model.sigma for p in self.RAM_physics.physics_list]))
        y_ram = dinv.utils.TensorList([
            utils.inverse_image_transform(y)*min/self.RAM_physics.physics_list[0].noise_model.sigma, 
            xt_init*min/self.RAM_physics.physics_list[1].noise_model.sigma,
            utils.inverse_image_transform(z)*min/self.RAM_physics.physics_list[2].noise_model.sigma,
        ]
        )
        x_hat = self.RAM(y_ram, self.RAM_physics)
        return utils.image_transform(x_hat)
      
    def run_unfolded_loop(
        self, 
        obs:dict,
        x:torch.Tensor, 
        t:int, 
        lambda_sig:float = 1.0,
        checkpoint = False,  
        collect_ims = False,
        **kwargs
    ) -> dict:
        self.obs = obs
        x_t, y = self.obs["x_t"], self.obs["y"]
        B,C = x.shape[:2]

        y_label = self.obs["y_label"] if "y_label" in self.obs.keys() else None
        IMSOUT = {}
        
        # x = x / models.extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t_denoising, x_t.shape).ravel()[0]
        for ii in range(self.max_unfolded_iter):
            ind_ii = ii % self.num_weights_hqs
            # parameters
            sigma_t = models.extract_tensor(self.diffusion_scheduler.sigmas, t, x_t.shape).ravel()[0]
            if self.args.learned_lambda_sig:
                lambda_sig = self.LambdaSIG[ind_ii].exp() 
            else:
                lambda_sig = self.args.lambda_
            rho, t_denoising, ty = self.get_denoising_timestep.get_z_timesteps(t, self.physic.sigma_model, x_t.shape, T_AUG = 1000*self.TAUG[ind_ii], sp = self.SP_Param[ind_ii].exp(), task=self.args.task)
            sqrt_alpha_comprod = models.extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape).ravel()[0]    
            
            self.params = {
                "sigma_model":self.physic.sigma_model, 
                "sigma_t":sigma_t, 
                "sqrt_alpha_comprod":sqrt_alpha_comprod,
                "rho":rho,
                "lambda_sig":lambda_sig,
                "ds_max_iter":5,
            }
            # --- initialisation
            out = self.proxf_update(x, t)
            zest = out["zest"]
            
            
            # noise prediction
            if self.args.pertub:
                noise = torch.randn_like(zest)
                val = torch.ones_like(t_denoising).long()
                max_val = 1000 * torch.ones_like(t_denoising).long()
                tu = torch.clamp(t_denoising + 1000*self.TLAT[ind_ii], val, max_val).long()
                z_input = self.diffusion_scheduler.sample_xt(zest, tu, noise=noise)
            else:
                z_input = zest
                tu = t_denoising.long()
                        
            # check if z_input is double
            z_input = z_input.float()

            # Denoising
            if checkpoint and False:
                def run_model(z, t):
                    return self.model(z, t, y=y_label)
                eps = torch.utils.checkpoint.checkpoint(run_model, z_input, tu, use_reentrant=False)
            else:
                eps = self.model(z_input, tu, y=y_label)
            eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_t.shape[2:]) else (eps, eps)
            
            # estimate of x_start
            x = self.diffusion_scheduler.predict_xstart_from_eps(z_input, tu, eps)
            if collect_ims:
                IMSOUT[f"x_{ii}"] = x
                IMSOUT[f"z_{ii}"] = z_input
                IMSOUT[f"eps_{ii}"] = eps

            x_0 = x.add(1).div(2)
            x_0 = torch.clamp(x_0, 0.0, 1.0)
           
            if self.args.task=="inp" and ii < self.max_unfolded_iter-2:
                x = (1-self.physic.Mask)*x + (self.physic.Mask)*y
        pred_eps = self.diffusion_scheduler.predict_eps_from_xstart(x_t, t, x)

        return {"xstart_pred":x_0, "aux":torch.clamp(zest.add(1).div(2),0,1), "z_input":z_input, "eps_pred":pred_eps,"eps_z":eps, "IMDICT":IMSOUT}
