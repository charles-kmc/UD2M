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


class Conditional_sampler:
    def __init__(
        self, 
        hqs_model:ums.HQS_models, 
        physics:Union[phy.Deblurring,phy.Inpainting,phy.SuperResolution], 
        diffusion:ums.DiffusionScheduler,
        device,
        args,
        max_unfolded_iter:int = 3,
        solver_type = "ddim",
        classic_sampling:bool=False
    ) -> None:
        self.hqs_model = hqs_model
        self.diffusion = diffusion
        self.physics = physics
        self.device = device
        self.args = args
        self.max_unfolded_iter = max_unfolded_iter
        self.solver_type = solver_type
        self.classic_sampling= classic_sampling
        
        # DPIR model
        self.dpir_model = DPIR_deb(
            device = self.device, 
            model_name = self.args.dpir.model_name, 
            pretrained_pth=self.args.dpir.pretrained_pth
        )
        
        # metric
        self.metrics = utils.Metrics(self.device)
    
    def sampler(
        self, 
        y:torch.tensor, 
        im_name:str, 
        num_timesteps:int = 100, 
        eta:float=0., 
        zeta:float=1.,
        x_true:torch.Tensor = None,
        lambda_:float=1
    ):
    
        with torch.no_grad():
            # sequence of timesteps
            seq, progress_seq = self.diffusion.get_seq_progress_seq(num_timesteps)
            diffusionsolver = DiffusionSolver(seq, self.diffusion)
            
            # initilisation
            if self.args.task=="deblur":
                if  self.args.dpir.use_dpir:
                    y_01 = utils.inverse_image_transform(y)
                    x0 = self.dpir_model.run(y_01, self.physics.blur, iter_num = 1) #y.clone()
                    x0 = utils.image_transform(x0)
                else:
                    x0 = y.clone()
            elif  self.args.task=="sr":
                if self.args.dpir.use_dpir:
                    y_ = self.physics.upsample(utils.inverse_image_transform(y))
                    y_ = self.dpir_model.run(y_, self.physics.blur, iter_num = 1)
                    x0 = utils.image_transform(y_)
                else:
                    y_ = self.physics.upsample(utils.inverse_image_transform(y))
                    x0 = utils.image_transform(y_)
            elif self.args.task=="inp":
                x0 = y.clone()
            elif self.args.task=="ct":
                x0=...
            else:
                raise ValueError(f"This task ({self.args.task})is not implemented!!!")
                
            # initilisation
            x = torch.randn_like(x0)  
            progress_img = []
            
            # reverse process
            if x_true is not None:
                psnrs = []
            N = len(seq)
            for ii in range(N):
                t_i = seq[ii]
                if self.classic_sampling:
                    out_val = self.sampling_prior(x, torch.tensor(t_i).to(self.device))
                    x0 = out_val["xstart_pred"].mul(2).add(-1)
                    
                else:
                    max_unfolded_iter = self.max_unfolded_iter
                    obs = {"x_t":x, "y":y}
                    if not self.args.task=="inp":
                        obs["blur"] = self.physics.blur
                    out_val = self.hqs_model.run_unfolded_loop(obs, x0, torch.tensor(t_i).to(self.device), max_iter = max_unfolded_iter, lambda_sig= lambda_)
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
                            out_val = self.hqs_model.run_unfolded_loop( y, x0, x, torch.tensor(t_i).to(self.device), max_iter = self.max_unfolded_iter)
                            x0 = utils.image_transform(out_val["xstart_pred"])
                            iim1 = ii + 1
                            x = diffusionsolver.huen_solver(x0, x, iim1, d_old=d_old)
                    else:
                        raise ValueError(f"This solver {self.solver_type} is not implemented. Please verify your solver.")
                else:
                    x = x0
    
                # save the process
                if self.args.save_progressive and (seq[ii] in progress_seq):
                    x_show = utils.get_rgb_from_tensor(utils.inverse_image_transform(x[0].unsqueeze(0)))      #[0,1]
                    progress_img.append(x_show)
                    utils.delete([x_show])
            
            if self.args.save_progressive:   
                img_total = cv2.hconcat(progress_img)
                if self.args.use_wandb and self.args.save_wandb_img:
                    image_wdb = wandb.Image(img_total, caption=f"progress sequence for im {im_name}", file_type="png")
                    wandb.log({"pregressive":image_wdb})
                    utils.delete([progress_img])
        out_val["xstart_pred"] = utils.inverse_image_transform(x)
        out_val["progress_img"] = img_total
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


class Conditional_sampler_olddddd:
    def __init__(
        self, 
        hqs_model:ums.HQS_models, 
        physics:Union[phy.Deblurring,phy.Inpainting,phy.SuperResolution], 
        diffusion:ums.DiffusionScheduler,
        device,
        args,
        max_unfolded_iter:int = 3,
        solver_type = "ddim",
        classic_sampling:bool = False,
    ) -> None:
        self.hqs_model = hqs_model
        self.diffusion = diffusion
        self.physics = physics
        self.device = device
        self.args = args
        self.max_unfolded_iter = max_unfolded_iter
        self.solver_type = solver_type
        self.classic_sampling = classic_sampling
        # DPIR model
        self.dpir_model = DPIR_deb(
            device = self.device, 
            model_name = self.args.dpir.model_name, 
            pretrained_pth=self.args.dpir.pretrained_pth
        )
        
        # metric
        self.metrics = utils.Metrics(self.device)
    
    def sampler(
        self, 
        y:torch.Tensor, 
        im_name:str, 
        num_timesteps:int = 100, 
        eta:float=0., 
        zeta:float=1.,
        x_true:torch.Tensor = None,
        track:bool = False,
        lambda_:int = 1
    ):
    
        with torch.no_grad():
            # sequence of timesteps
            seq, progress_seq = self.diffusion.get_seq_progress_seq(num_timesteps)
            diffusionsolver = DiffusionSolver(seq, self.diffusion)
    
            # initilisation
            if self.args.dpir.use_dpir and self.args.task=="deblur":
                y_01 = utils.inverse_image_transform(y)
                x0 = self.dpir_model.run(y_01, self.physics.blur, iter_num = 1) #y.clone()
                x0 = utils.image_transform(x0)
            elif  self.args.task=="sr":
                if self.args.dpir.use_dpir:
                    y_ = self.physics.upsample(utils.inverse_image_transform(y))
                    y_ = self.dpir_model.run(y_, self.physics.blur, iter_num = 1)
                    x0 = utils.image_transform(y_)
                else:
                    y_ = self.physics.upsample(utils.inverse_image_transform(y))
                    x0 = utils.image_transform(y_)
            else:
                x0 = y.clone()
                
            x = torch.randn_like(x0)
            progress_img = []
            
            # reverse process
            if x_true is not None:
                psnrs = []
            for ii in range(len(seq)):
                t_i = seq[ii]
                if self.classic_sampling:
                    out_val = self.sampling_prior(x, torch.tensor(t_i).to(self.device))
                    x0 = utils.image_transform(out_val["xstart_pred"])
                else:
                    # unfolding: xstrat_pred, eps_pred, auxiliary variable
                    max_unfolded_iter = self.max_unfolded_iter
                    
                    obs = {"x_t":x, "y":y}
                    if not self.args.task=="inp":
                        obs["blur"] = self.physics.blur
                    out_val = self.hqs_model.run_unfolded_loop(obs, x0, torch.tensor(t_i).to(self.device), max_iter = max_unfolded_iter, lambda_sig= lambda_)
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
                    else:
                        raise ValueError(f"This solver {self.solver_type} is not implemented. Please verify your solver.")
                else:
                    x = x0
    
                # save the process
                if self.args.save_progressive and (seq[ii] in progress_seq):
                    x_show = utils.get_rgb_from_tensor(utils.inverse_image_transform(x))      #[0,1]
                    progress_img.append(x_show)
                    utils.delete([x_show])
            
            if self.args.save_progressive:   
                img_total = cv2.hconcat(progress_img)
                if self.args.use_wandb and self.args.save_wandb_img:
                    image_wdb = wandb.Image(img_total, caption=f"progress sequence for im {im_name}", file_type="png")
                    wandb.log({"pregressive":image_wdb})
                    utils.delete([progress_img])
        
        out_val["progress_img"] = img_total
        if x_true is not None:
            out_val["psnrs"] = psnrs
        return out_val
        
    def sampling_prior(self, xt, t):
        B,C = xt.shape[:2]
        pred_eps = self.model(xt, t)
        pred_eps, _ = torch.split(pred_eps, C, dim=1) if pred_eps.shape == (B, C * 2, *xt.shape[2:]) else (pred_eps, pred_eps)
        
        # estimate of x_start
        x = self.diffusion.predict_xstart_from_eps(xt, t, pred_eps)
        x_pred = utils.inverse_image_transform(x)
        x0 = torch.clamp(x_pred, 0.0, 1.0)
    
        return {"xstart_pred":x0, "aux":x0, "eps_pred":pred_eps}

       