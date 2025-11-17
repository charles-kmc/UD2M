import torch 
import numpy as np
import cv2
import wandb
from typing import Union

from .runners_utils import DiffusionSolver
from DPIR.dpir_models import DPIR_deb
import unfolded_models as ums
import physics as phy
import utils as utils
import models as models
import deepinv as dinv
from unfolded_models.unfolded_models import StackedJointPhysics

class Conditional_sampler:
    def __init__(
        self, 
        hqs_model:ums.HQS_models, 
        physics:Union[phy.Deblurring,phy.Inpainting,phy.SuperResolution, phy.JPEG], 
        diffusion:ums.DiffusionScheduler,
        device,
        args,
        max_unfolded_iter:int = 3,
        solver_type = "ddim",
        classic_sampling:bool=False,
        dphys = None,
        img_shape = (3,256,256),
        state_dict_RAM:dict = None
    ) -> None:
        self.hqs_model = hqs_model
        self.diffusion = diffusion
        self.physics = physics
        self.device = device
        self.args = args
        self.max_unfolded_iter = max_unfolded_iter
        self.solver_type = solver_type
        self.classic_sampling= classic_sampling
        self.d_phys = dphys
        self.img_shape = img_shape

        if dphys is not None:
            from ram import RAM
            self.RAM = RAM(device=device)
            if state_dict_RAM is not None:
                self.RAM.load_state_dict(state_dict_RAM, strict=False)
            self.RAM.to(self.device)
            self.RAM.eval()
            for k, v in self.RAM.named_parameters():
                v.requires_grad = False
            self.stacked_physic = StackedJointPhysics(
                self.d_phys, self.diffusion
            )
        
        # DPIR model
        if self.args.dpir.use_dpir:
            self.dpir_model = DPIR_deb(
                device = self.device, 
                model_name = self.args.dpir.model_name, 
                pretrained_pth=self.args.dpir.pretrained_pth
            )
        
        # metric
        self.metrics = utils.Metrics(self.device)
    
    def RAM_Initialization(self, y, xt, t):
        if isinstance(t, torch.Tensor) and t.numel()>1:
            t = int(t[0].item())
        self.stacked_physic.set_t(t)
        
        xt_init = (xt.clone() + self.stacked_physic.physics_list[1].alphas_cumprod[t].sqrt())/2 
        min = torch.min(torch.tensor([p.noise_model.sigma for p in self.stacked_physic.physics_list]))
        y_ram = dinv.utils.TensorList([utils.inverse_image_transform(y)*min/self.d_phys.noise_model.sigma, xt_init*min/self.stacked_physic.physics_list[1].noise_model.sigma])
        x_init = self.RAM(y_ram, self.stacked_physic)
        return utils.image_transform(x_init)
    
    def sampler(
        self, 
        y:torch.tensor, 
        im_name:str, 
        num_timesteps:int = 100, 
        eta:float=0., 
        zeta:float=1.,
        x_true:torch.Tensor = None,
        lambda_:float=1,
        y_label = None
    ):
    
        with torch.no_grad():
            # sequence of timesteps
            seq, progress_seq = self.diffusion.get_seq_progress_seq(num_timesteps)
            diffusionsolver = DiffusionSolver(seq, self.diffusion)
            
            # initilisation
            y_ = utils.inverse_image_transform(y)
            if self.d_phys is not None:
                x_init = self.RAM_Initialization(y, torch.randn(self.img_shape).to(self.device), torch.tensor(self.diffusion.num_timesteps-1).to(self.device))
                
            elif self.args.task=="deblur":
                if  self.args.dpir.use_dpir:
                    x_init = self.dpir_model.run(y_, self.physics.blur, iter_num = 1)
                    x_init = utils.image_transform(x_init)
                else:
                    x_init = y.clone()
                    
            elif self.args.task=="sr":
                if self.args.dpir.use_dpir:
                    y_ = self.physics.upsample(y_)
                    x_init = self.dpir_model.run(y_, self.physics.blur, iter_num = 1)
                    x_init = utils.image_transform(x_init)
                else:
                    x_init = self.physics.AT(self.physics.upsample(y))
            elif self.args.task=="inp":
                x_init = y.clone()
            else:
                raise ValueError(f"This task ({self.args.task})is not implemented!!!")
            
            # initilisation
            x = torch.randn_like(x_init)  
            progress_img = []
            progress_zero = []
            
            # reverse process
            if x_true is not None:
                psnrs = []
            N = len(seq)
            for ii in range(N):
                t_i = seq[ii]
                rt_alpha = self.diffusion.sqrt_alphas_cumprod[t_i].reshape(-1,1,1,1)
                sigma_t = (1.0-rt_alpha**2).sqrt()
                if self.args.task=="sr":
                    x_init = (self.physics.AT(self.physics.upsample(y))*sigma_t + rt_alpha*x*self.physics.sigma_model)/(self.physics.sigma_model**2 + sigma_t**2).sqrt()
                elif self.args.task in ["deblur","inp"]:
                    x_init = (self.physics.AT(y)*sigma_t + rt_alpha*x*self.physics.sigma_model)/(self.physics.sigma_model**2 + sigma_t**2).sqrt()
                if self.classic_sampling:
                    out_val = self.sampling_prior(x, torch.tensor(t_i).to(self.device))
                    x0 = out_val["xstart_pred"].mul(2).add(-1)
                else:
                    max_unfolded_iter = self.max_unfolded_iter
                    obs = {"x_t":x, "y":y, "y_label":y_label}
                    if not self.args.task=="inp":
                        obs["blur"] = self.physics.blur
                    # Modified12 May 25 - always initialize HQS with the same x0 (xt appears through the prox step)
                    out_val = self.hqs_model.run_unfolded_loop(obs, x_init, torch.tensor(t_i).to(self.device), max_iter = max_unfolded_iter, lambda_sig= lambda_)
                    x0 = out_val["xstart_pred"].mul(2).add(-1)
                        
                if x_true is not None:
                    psnrs.append(self.metrics.psnr_function(x_true, out_val["xstart_pred"]).cpu().numpy())
                    if self.args.use_wandb:
                        wandb.log({f"progression of the psnr in the reverse process {im_name}":self.metrics.psnr_function(x_true, out_val["xstart_pred"])})
                
                # sample from the predicted noise
                if seq[ii] != seq[-1] and ii < len(seq)-1: 
                    if self.solver_type == "ddim":
                        x = diffusionsolver.ddim_solver(x0, out_val["eps_pred"], ii, eta, zeta)
                    elif self.solver_type == "ddpm":
                        x = diffusionsolver.ddpm_solver(x, out_val["eps_pred"], ii, ddpm_param = self.args.ddpm_param)
                    elif self.solver_type == "huen":
                        x, d_old = diffusionsolver.huen_solver(x, out_val["eps_pred"], ii)
                        tiim1 = self.seq[ii + 1]  
                        beta_tiim1 = models.extract_tensor(self.diffusion.betas, tiim1, x.shape)
                        if beta_tiim1 !=0:
                            out_val = self.hqs_model.run_unfolded_loop(y, x0, x, torch.tensor(t_i).to(self.device), max_iter = self.max_unfolded_iter)
                            x0 = utils.image_transform(out_val["xstart_pred"])
                            iim1 = ii + 1
                            x = diffusionsolver.huen_solver(x0, x, iim1, d_old=d_old)
                    else:
                        raise ValueError(f"This solver {self.solver_type} is not implemented. Please verify your solver.")
                else:
                    x = x0
    
                if self.d_phys is not None:
                    x_init = self.RAM_Initialization(y, x, torch.tensor(t_i).to(self.device))
                else:
                    x_init=x0.clone()  # save x_init for the next iteration
                    
                # save the process
                if self.args.save_progressive and (seq[ii] in progress_seq):
                    x_show = utils.get_rgb_from_tensor(utils.inverse_image_transform(x[0].unsqueeze(0)))      #[0,1]
                    x_0show = utils.get_rgb_from_tensor(utils.inverse_image_transform(x0[0].unsqueeze(0)))      #[0,1]
                    progress_img.append(x_show)
                    progress_zero.append(x0[0])
                    utils.delete([x_show])
            
            if self.args.save_progressive:   
                img_total = cv2.hconcat(progress_img)
                # img_zero_total = cv2.hconcat(progress_zero)
                if self.args.use_wandb and self.args.save_wandb_img:
                    image_wdb = wandb.Image(img_total, caption=f"progress sequence for im {im_name}", file_type="png")
                    wandb.log({"pregressive":image_wdb})
                    utils.delete([progress_img])
                    
        out_val["xstart_pred"] = utils.inverse_image_transform(x)
        out_val["progress_img"] = img_total
        out_val["progress_zero"] = progress_zero
        out_val["x_init"] = utils.inverse_image_transform(x_init)
        if x_true is not None:
            out_val["psnrs"] = psnrs
        return out_val
    
    def sampling_prior(self, xt, t):
        B,C = xt.shape[:2]
        pred_eps = self.hqs_model.model(xt, t)
        pred_eps, _ = torch.split(pred_eps, C, dim=1) if pred_eps.shape == (B, C * 2, *xt.shape[2:]) else (pred_eps, pred_eps)
        
        # estimate of x_start
        x = self.diffusion.predict_xstart_from_eps(xt, t, pred_eps)
        x_pred = utils.inverse_image_transform(x)
        x0 = torch.clamp(x_pred, 0.0, 1.0)
    
        return {"xstart_pred":x0, "aux":x0, "eps_pred":pred_eps}
