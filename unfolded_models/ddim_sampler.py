import torch.nn as nn
from torch.utils.data import DataLoader
import torch 
import numpy as np
import datetime
import math
import os 
import cv2
import wandb
import copy
import yaml
import blobfile as bf

from utils.utils import Metrics, inverse_image_transform, image_transform, delete
from DPIR.dpir_models import DPIR_deb
from args_unfolded_model import args_unfolded
from utils_lora.lora_parametrisation import LoRa_model
from datasets.datasets import Datasets
from .unfolded_model import (
    extract_tensor,
    get_rgb_from_tensor,
    DiffusionScheduler,
    physics_models, 
    HQS_models,
    GetDenoisingTimestep,
)



class DDIM_SAMPLER:
    def __init__(
        self, 
        hqs_model:HQS_models, 
        physics:physics_models, 
        diffusion:DiffusionScheduler,
        device,
        args,
        max_unfolded_iter:int = 3,
        solver_type = "ddim",
    ) -> None:
        self.hqs_model = hqs_model
        self.diffusion = diffusion
        self.physics = physics
        self.device = device
        self.args = args
        self.max_unfolded_iter = max_unfolded_iter
        self.solver_type = solver_type
        
        # DPIR model
        self.dpir_model = DPIR_deb(
            device = self.device, 
            model_name = self.args.dpir.model_name, 
            pretrained_pth=self.args.dpir.pretrained_pth
        )
        
        # metric
        self.metrics = Metrics(self.device)

        # Huen sampler 
        huen_module = Huen_SAMPLER()
    def sampler(
        self, 
        y:torch.tensor, 
        im_name:str, 
        num_timesteps:int = 100, 
        eta:float=1, 
        zeta:float=1,
        x_true = None
    ):
    
        with torch.no_grad():
            # sequence of timesteps
            seq, progress_seq = self.diffusion.get_seq_progress_seq(num_timesteps)
            diffusionsolver = DiffusionSolver(seq, self.diffusion)
            # Huen sampler 
            huen_module = Huen_SAMPLER(seq, device=self.device)
            
            # initilisation
            x = torch.randn_like(y)

            if self.args.dpir.use_dpir:
                y_ = inverse_image_transform(y)
                x0 = self.dpir_model.run(y_, self.physics.blur_kernel, iter_num = 1) #y.clone()
                x0 = image_transform(x0)
            else:
                x0 = y.clone()
                
            progress_img = []
            
            # reverse process
            if x_true:
                psnrs = []
            N = len(seq)
            for ii in range(N):
                t_i = seq[ii]
                # unfolding: xstrat_pred, eps_pred, auxiliary variable
                out_val = self.hqs_model.run_unfolded_loop( y, x0, x, torch.tensor(t_i).to(self.device), max_iter = self.max_unfolded_iter)
                x0 = image_transform(out_val["xstart_pred"])
                if x_true:
                    psnrs.append(self.metrics.psnr_function(x_true, out_val["xstart_pred"]))
                
                # sample from the predicted noise
                if seq[ii] != seq[-1] and ii < len(seq)-1: 
                    if self.solver_type == "ddim":
                        x = diffusionsolver.ddim_solver(x0, out_val["eps_pred"], ii, eta, zeta)
                    elif self.solver_type == "ddpm":
                        x = diffusionsolver.ddpm_solver(x, out_val["eps_pred"], ii, ddpm_param = self.args.ddpm_param)
                    elif self.solver == "huen":
                        x, d_old = diffusionsolver.huen_solver(x, out_val["eps_pred"], ii) 
                        
                        tiim1 = self.seq[ii + 1]  
                        beta_tiim1 = extract_tensor(self.diffusion.betas, tiim1, x.shape)
                        if beta_tiim1 !=0:
                            out_val = self.hqs_model.run_unfolded_loop( y, x0, x, torch.tensor(t_i).to(self.device), max_iter = self.max_unfolded_iter)
                            x0 = image_transform(out_val["xstart_pred"])
                            iim1 = ii + 1
                            x = diffusionsolver.huen_solver(x0, x, iim1, d_old=d_old)
                    else:
                        raise ValueError(f"This solver {self.solver_type} is not implemented. Please verify your solver.")
                else:
                    x = x0
    
                # save the process
                if self.args.save_progressive and (seq[ii] in progress_seq):
                    x_show = get_rgb_from_tensor(inverse_image_transform(x))      #[0,1]
                    progress_img.append(x_show)
                    delete([x_show])
            
            if self.args.save_progressive:   
                img_total = cv2.hconcat(progress_img)
                if self.args.use_wandb and self.args.save_wandb_img:
                    image_wdb = wandb.Image(img_total, caption=f"progress sequence for im {im_name}", file_type="png")
                    wandb.log({"pregressive":image_wdb})
                    delete([progress_img])
        
        out_val["progress_img"] = img_total
        if x_true:
            out_val["psnrs"] = psnrs
        return out_val

class DiffusionSolver:
    def __init__(
        self, 
        seq, 
        diffusion:DiffusionScheduler, 
    ):
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
        beta_t = extract_tensor(self.diffusion.betas, t_ii, x_t.shape)
        return (x_tiim1, d_new) if d_old is None else x_tiim1
    
    
class Huen_SAMPLER:
    def __init__(self, seq, device = "cpu", sigma_min=0.002, sigma_max=80, num_timesteps=1000, rho = 7):
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_timesteps = num_timesteps
        self.device = device
        self.seq=seq
        
        t = torch.linspace(0, self.num_timesteps, self.num_timesteps).to(self.device)
        
        self.const1 = self.sigma_min**(1/self.rho) 
        self.const2 = (-self.sigma_min**(1/self.rho)+self.sigma_max**(1/self.rho))/(self.num_timesteps-1)
        self.sigma_ts = (self.const1 + t * self.const2)**self.rho
        self.sigma_prime_ts = torch.ones_like(t)
    
    #predict x_start from eps 
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {eps.shape})")
        return (
            (x_t - extract_tensor( self.sigma_ts, t, x_t.shape ) * eps) 
        )
        
    def _linearScheduler(self):
        scale = 1000 / self.num_timesteps
        sigma_start = scale * self.sigma_min
        sigma_end = scale * self.sigma_max
        sigmas = np.linspace(sigma_start, sigma_end, self.num_timesteps, dtype=np.float32)
        return sigmas
    
    # get sequence of timesteps 
    def get_seq_progress_seq(self, iter_num):
        seq = np.sqrt(np.linspace(0, self.num_timesteps**2, iter_num))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1
        progress_seq = seq[::max(len(seq)//10,1)]
        if progress_seq[-1] != seq[-1]:
            progress_seq.append(seq[-1])
        seq = seq[::-1]
        progress_seq = progress_seq[::-1]
        return (seq, progress_seq)
    
    
    # TODO replace ii with tii 
    def huen_solver(self, x_tii, eps, ii, d_old=None, scaling=None):
        tii = self.seq[ii]
        sigma_tii = extract_tensor(self.sigma_ts,tii, x_tii.shape)
        sigma_prime_tii = extract_tensor(self.sigma_prime_ts,tii, x_tii.shape)
        x_pred = self.predict_xstart_from_eps(x_tii, tii, eps)
        
        d_new = (sigma_prime_tii/sigma_tii)*(x_tii - x_pred)
        if d_old is not None:
            d_new = 0.5*(d_new + d_old) 
        tiim1 = self.seq[ii+1]
        if scaling is None:
            x_tiim1 = x_tii + (tiim1 - tii)*d_new
        else:
            x_tiim1 = x_tii + scaling*d_new
        tiim1 = self.seq[ii+1]
    
        if d_old is None and scaling is None:
            return x_tiim1, d_new, (tiim1 - tii)
        else: 
            return x_tiim1
        
    def huen_solver(self, x_tii, eps, ii, d_old=None, scaling=None):
        tii = self.seq[ii]
        sigma_tii = extract_tensor(self.sigma_ts,tii, x_tii.shape)
        sigma_prime_tii = extract_tensor(self.sigma_prime_ts,tii, x_tii.shape)
        x_pred = self.predict_xstart_from_eps(x_tii, tii, eps)
        
        d_new = (sigma_prime_tii/sigma_tii)*(x_tii - x_pred)
        if d_old is not None:
            d_new = 0.5*(d_new + d_old) 
        tiim1 = self.seq[ii+1]
        if scaling is None:
            x_tiim1 = x_tii + (tiim1 - tii)*d_new
        else:
            x_tiim1 = x_tii + scaling*d_new
        tiim1 = self.seq[ii+1]
    
        if d_old is None and scaling is None:
            return x_tiim1, d_new, (tiim1 - tii)
        else: 
            return x_tiim1
# load lara weights                
def load_trainable_params(model, epoch, args):
    checkpoint_dir = bf.join(args.path_save, args.task, "Checkpoints", args.date)
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

