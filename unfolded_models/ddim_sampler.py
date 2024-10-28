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
from models.load_frozen_model import load_frozen_model
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
    ) -> None:
        self.hqs_model = hqs_model
        self.diffusion = diffusion
        self.physics = physics
        self.device = device
        self.args = args
        self.max_unfolded_iter = max_unfolded_iter
        
        # DPIR model
        self.dpir_model = DPIR_deb(
            device = self.device, 
            model_name = self.args.dpir.model_name, 
            pretrained_pth=self.args.dpir.pretrained_pth
        )
        
        # metric
        self.metrics = Metrics(self.device)
    
    def sampler(
        self, 
        y:torch.tensor, 
        im_name:str, 
        num_timesteps:int = 100, 
        eta:float=1, 
        zeta:float=1
    ):
    
        with torch.no_grad():
            # sequence of timesteps
            seq, progress_seq = self.diffusion.get_seq_progress_seq(num_timesteps)
            
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
            for ii in range(len(seq)):
                t_i = seq[ii]
                # unfolding: xstrat_pred, eps_pred, auxiliary variable
                out_val = self.hqs_model.run_unfolded_loop( y, x0, x, torch.tensor(t_i).to(self.device), max_iter = self.max_unfolded_iter)
                x0 = image_transform(out_val["xstart_pred"])
                
                # sample from the predicted noise
                if seq[ii] != seq[-1] and ii < len(seq)-1: 
                    t_im1 = seq[ii+1]
                    beta_t = self.diffusion.betas[seq[ii]]
                    eta_sigma = eta * self.diffusion.sqrt_1m_alphas_cumprod[t_im1] / self.diffusion.sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(beta_t)
                    x = self.diffusion.sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(self.diffusion.sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * out_val["eps_pred"] \
                                    + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * self.diffusion.sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                
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
        
        return out_val

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

