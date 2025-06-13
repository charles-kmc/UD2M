import torch
from typing import Union
import wandb

from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep
import utils as utils
import physics as phy
import models as models
import deepinv as dinv
from runners.runners_utils import DiffusionSolver

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
class DiffPIRModules(torch.nn.Module):
    def __init__(
        self,
        model, 
        physic:Union[phy.Deblurring, phy.Inpainting, phy.SuperResolution], 
        diffusion_scheduler:DiffusionScheduler, 
        get_denoising_timestep:GetDenoisingTimestep,
        args,
        device,
        max_iter:int = 3,
        iter_dep_weights=False,
        cm = True,
        **kwargs
        ) -> None:
        super().__init__()
        self.cm = cm
        self.model = model
        self.max_iter = max_iter
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
        self.get_denoising_timestep = get_denoising_timestep
        self.args = args
        self.num_weights_hqs = self.max_iter if iter_dep_weights else 1
        self.SP_Param = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(self.num_weights_hqs)])
        self.LambdaSIG= torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(self.num_weights_hqs)])  # Used to augment the denoising step
        self.TAUG = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(self.num_weights_hqs)])  # Used to augment the denoising step
        self.TLAT = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.0, device=device), requires_grad=True) for _ in range(self.num_weights_hqs)])  # Used to auhment noise added before the denosiing step

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
        lambda_sig:float = 1.0,
        checkpoint = True,
        collect_ims = False,
        **kwargs
    ) -> dict:
        self.obs = obs
        x_tau, y = self.obs["x_t"], self.obs["y"]
        B,C = x_tau.shape[:2]

        y_label = self.obs["y_label"] if "y_label" in self.obs.keys() else None
        IMSOUT = {}
        
        seq, _ = self.diffusion_scheduler.get_seq_progress_seq(self.max_iter, n_times = t)
        for ii, tau in enumerate(seq):
            tau = torch.tensor(tau, device=x_tau.device).unsqueeze(0).long()
            ind_ii = ii % self.num_weights_hqs
            # parameters
            sigma_t = models.extract_tensor(self.diffusion_scheduler.sigmas, tau, x_tau.shape).ravel()[0]
            lambda_sig = self.LambdaSIG[ind_ii].exp() 
            rho, t_denoising, ty = self.get_denoising_timestep.get_z_timesteps(tau, self.physic.sigma_model, x_tau.shape, T_AUG = 0, sp = self.SP_Param[ind_ii].exp())
            
            sqrt_alpha_comprod = models.extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, tau, x_tau.shape).ravel()[0]    
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


            # Denoising   
            if self.cm:
                if checkpoint:
                    def run_model(z, t):
                        return self.model(z, t, y = y_label)
                    x = torch.utils.checkpoint.checkpoint(run_model, x_tau, tau, use_reentrant=False)
                else:
                    x = self.model(x_tau, tu, y=y_label)
                eps = self.diffusion_scheduler.predict_eps_from_xstart_cont(x_tau, tau, x0)
            else:
                if checkpoint:
                    def run_model(z, t):
                        return self.model(z, t, y = y_label)
                    eps = torch.utils.checkpoint.checkpoint(run_model, x_tau, tau, use_reentrant=False)
                else:
                    eps = self.model(x_tau, tu, y=y_label)
                eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_tau.shape[2:]) else (eps, eps)
                
                # estimate of x_start
                x = self.diffusion_scheduler.predict_xstart_from_eps(x_tau, tau, eps)
            zest = x
            x = self.proxf_update(x, t)
            eps = self.diffusion_scheduler.predict_eps_from_xstrat(x_tau, tau, x["zest"])
            if collect_ims:
                IMSOUT[f"x_{ii}"] = x["zest"]
                IMSOUT[f"z_{ii}"] = x_tau
                IMSOUT[f"eps_{ii}"] = eps

            # DDIM Step 
            if ii < self.max_iter - 1:
                x_tau = DiffusionSolver(seq=seq, diffusion=self.diffusion_scheduler).ddim_solver(
                    x0=x["zest"], 
                    eps=eps, 
                    ii=ii, 
                    eta=self.TAUG[ind_ii].exp(), 
                    zeta=self.TLAT[ind_ii].exp()
                )

            x_0 = x["zest"].add(1).div(2)
            x_0 = torch.clamp(x_0, 0.0, 1.0)
            
            if self.args.task=="inp" and ii<self.max_iter-1:
                x = (1-self.physic.Mask)*x["zest"] + (self.physic.Mask)*y
        
        pred_eps = self.diffusion_scheduler.predict_eps_from_xstrat(self.obs["x_t"], t, x["zest"])

        return {"xstart_pred":x_0, "aux":torch.clamp(zest.add(1).div(2),0,1), "z_input":x_tau, "eps_pred":pred_eps,"eps_z":eps, "IMDICT":IMSOUT}
