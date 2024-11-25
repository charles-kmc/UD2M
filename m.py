import os
import copy
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import datetime
import flags
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from unfolded_models.unfolded_model import (
    HQS_models, 
    physics_models, 
    DiffusionScheduler,
    GetDenoisingTimestep,
    get_rgb_from_tensor,
)
from unfolded_models.runner_inference_cmd import (
    load_trainable_params,
    DDIM_SAMPLER,
    load_yaml,
)
from args_unfolded_model import args_unfolded
from models.load_frozen_model import load_frozen_model
from datasets.datasets import Datasets
from utils_lora.lora_parametrisation import LoRa_model
from utils.utils import ( 
    inverse_image_transform, 
    image_transform,
    DotDict,
    welford,
    save_images,
    delete,
)

from metrics.coverage.coverage_function import coverage_eval
from metrics.sliced_wasserstein_distance.evaluate_swd import eval_swd
from metrics.metrics import Metrics
from metrics.fid_batch.fid_evaluation import Fid_evatuation

LAMBDA =np.arange(0.26,0.36, 0.02)
# LAMBDA = np.arange(1,10, 1)
stochastic_model = True
stochastic_rate = 0.05
LORA = True
NUM_TIMESTEPS = [400,1000]
eta = 0.0 
T_AUG = 10
max_iter = 1
ckpt_epoch = 840

with torch.no_grad():
    # args
    args = args_unfolded()
    args.lora = LORA
    args.ddpm_param = 1
    # parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    args.solver_type = "ddim"
    args.eta = 0.0
    args.zeta = 0.6
    args.max_unfolded_iter = 3
    args.mode = "eval"
    args.date = "17-11-2024"
    args.epoch = ckpt_epoch
    args.eval_date = "20-11-2024"
    args.save_wandb_img = False 
    args.stochastic_rate = stochastic_rate
             
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_name_folder = os.path.join(f"zeta_{args.zeta}", f"eta_{args.eta}")
    
    for num_timestep in NUM_TIMESTEPS:
        dic_results = {}
        vec_fid_mmse = []
        vec_psnr_mmse = []
        vec_ssim_mmse = []
        vec_lpips_mmse = []
        vec_fid_last = []
        for lambda_ in LAMBDA:
            
            root_dir = os.path.join(args.path_save, args.task, "Results_new", args.eval_date, "ddim", "lora", f"Max_iter_{max_iter}", f"unroll_iter_{args.max_unfolded_iter}", f"timesteps_{num_timestep}", f"Epoch_{args.epoch}", f"T_AUG_{T_AUG}", param_name_folder, f"stochastic_{args.stochastic_rate}", f"lambda_{lambda_}")
            save_metric_path = os.path.join(args.path_save, args.task, "Results_new", args.eval_date, "ddim", "lora", f"Max_iter_{max_iter}", f"unroll_iter_{args.max_unfolded_iter}", f"timesteps_{num_timestep}", f"Epoch_{args.epoch}", f"T_AUG_{T_AUG}", param_name_folder, f"stochastic_{args.stochastic_rate}", "metrics_evaluated")
            root_dir_csv = os.path.join(root_dir, "metrics_results")
            # FID
            out= Fid_evatuation(root_dir, device)
            vec_fid_mmse.append(out["fid_mmse"])
            vec_fid_last.append(out["fid_last"])
            dic_results[f"fid mmse_timesteps{lambda_}"] = [out["fid_mmse"]]
            dic_results[f"fid last_timesteps{lambda_}"] = [out["fid_last"]]
            
            # other metrics: PSNR, SSIM, LPIPs
            data = pd.read_csv(os.path.join(root_dir_csv, "metrics_results.csv"))
            mean_psnr = data["psnr"].mean()
            mean_lpips = data["lpips"].mean()
            mean_ssim = data["ssim"].mean()
            dic_results[f"psnr mmse_timesteps{lambda_}"] = [mean_psnr]
            dic_results[f"lpips mmse_timesteps{lambda_}"] = [mean_lpips]
            dic_results[f"ssim mmse_timesteps{lambda_}"] = [mean_ssim]
            vec_lpips_mmse.append(mean_lpips)
            vec_psnr_mmse.append(mean_psnr)
            vec_ssim_mmse.append(mean_ssim)
        
        # save data in a csv file 
        lab ="lambda"
        os.makedirs(save_metric_path, exist_ok=True)
        fid_pd = pd.DataFrame(dic_results)
        save_dir_metrics = os.path.join(save_metric_path, f'fid_psnr_lpips_ssim_results.csv') #_eta_{eta}_zeta_{zeta}
        fid_pd.to_csv(save_dir_metrics, mode='a', header=not os.path.exists(save_dir_metrics))
        
        # plot the results
        xaxis = LAMBDA
        plot = plt.figure()
        plt.plot(np.array(xaxis), np.array(vec_fid_mmse), "b.", label = "FID - mmse")
        plt.plot(np.array(xaxis), np.array(vec_fid_last), "r.", label = "FID - last")
        plt.xlabel(f"{lab}")
        plt.ylabel("FID")
        plt.title(f"{lab} versus FID")
        plt.legend()
        plt.savefig(os.path.join(save_metric_path, f"fid_results.png"))
        plt.close("all")
        
        # plot the results
        plot = plt.figure()
        plt.plot(np.array(xaxis), np.array(vec_lpips_mmse), "b.", label = "lpips")
        plt.legend()
        plt.xlabel(f"{lab}")
        plt.ylabel("LPIPs")
        plt.title(f"{lab} versus lpips")
        plt.savefig(os.path.join(save_metric_path, f"lpips_results.png")) # _eta_{eta}_zeta{zeta}
        plt.close("all")
        
        # plot the results
        plot = plt.figure()
        plt.plot(np.array(xaxis), np.array(vec_psnr_mmse), "b.", label = "psnr")
        plt.legend()
        plt.xlabel(f"{lab}")
        plt.ylabel("PSNR")
        plt.title("zeta versus psnr")
        plt.savefig(os.path.join(save_metric_path, f"psnr_results.png")) # _eta_{eta}_zeta{zeta}
        plt.close("all")
        
        # plot the results
        plot = plt.figure()
        plt.plot(np.array(xaxis), np.array(vec_ssim_mmse), "b.", label = "ssim")
        plt.legend()
        plt.xlabel(f"{lab}")
        plt.ylabel("SSIM")
        plt.title(f"{lab} versus ssim")
        plt.savefig(os.path.join(save_metric_path, f"ssim_results.png")) #_eta_{eta}_zeta{zeta}
        plt.close("all")
    