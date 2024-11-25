import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import DotDict


from metrics.fid_batch.fid_evaluation import Fid_evatuation

ZETA = [0.4]
TIMESTEPS = [200]#[20, 50, 100, 150, 200, 250, 350, 500, 600, 700]#, 800, 900, 1000]
def main_metrics(args, solver, axis_vec = ZETA, val = TIMESTEPS, val_name = "T", axis_name = "Zeta"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date = args.date_eval
    epoch = args.epoch
    max_iter = args.max_iter
    task = args.task 
    max_unfolded_iter = args.max_unfolded_iter 
    folder = "last"
    dic_results = {}
    vec_fid_mmse = []
    vec_psnr_mmse = []
    vec_ssim_mmse = []
    vec_lpips_mmse = []
 
    for axis in axis_vec:
        if axis_name == "Zeta":
            zeta = axis
            timesteps = val
        elif axis_name == "T":
            zeta = val
            timesteps = axis
        else:
            raise ValueError("Not implemented !!")
        # root_dir = f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/{task}/Results/{date}/{solver}/no_lora/Max_iter_{max_iter}/unroll_iter_{max_unfolded_iter}/timesteps_{timesteps}/Epoch_{epoch}/T_AUG_{T_AUG}/ddpm_param_3" #zeta_{zeta}/eta_{eta}
        # root_dir0 = f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/11-11-2024/ddim/lora/Max_iter_50/unroll_iter_3/timesteps_{timesteps}/Epoch_600/T_AUG_4/zeta_{zeta}/eta_0.0" #zeta_{zeta}/eta_{eta}
        # save_metric_path0 = os.path.join(f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/11-11-2024/ddim/lora/Max_iter_50/unroll_iter_3", "metrics_results_calib_timesteps_last")
        save_metric_path1= os.path.join("sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/20-11-2024/ddim/lora/Max_iter_20/unroll_iter_3,metrics_results_calib_timesteps_last")
        root_dir1= "/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/20-11-2024/ddim/lora/Max_iter_20/unroll_iter_3/timesteps_100/Epoch_600/T_AUG_10/zeta_0.4/eta_0.0/stochastic_0.05"
        root_dir2 = "/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/19-11-2024/ddim/lora/Max_iter_50/unroll_iter_3/timesteps_100/Epoch_600/T_AUG_10/zeta_0.4/eta_0.0"
        save_metric_path2 = os.path.join("sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/19-11-2024/ddim/lora/Max_iter_50/unroll_iter_3,metrics_results_calib_timesteps_last")
        root_dir = root_dir1
        save_metric_path = save_metric_path1
        # FID
        out = Fid_evatuation(root_dir, device, mmse_sample=True, last_sample=True)
        vec_fid_mmse.append(out["fid_last"])
        dic_results[f"fid mmse_vec_{axis_name}_{axis}"] = [out["fid_last"]]
        
        print(f"fid_last = {out['fid_last']}")
        print(f"fid_mmse = {out['fid_mmse']}")
       
        # other metrics: PSNR, SSIM, LPIPs
        root_dir_csv = os.path.join(root_dir, "metrics_results")
        data = pd.read_csv(os.path.join(root_dir_csv, "metrics_results.csv"))
        mean_psnr = data["psnr"].mean()
        mean_lpips = data["lpips"].mean()
        mean_ssim = data["ssim"].mean()
        dic_results[f"psnr mmse_{axis_name}_{axis}"] = [mean_psnr]
        dic_results[f"lpips mmse_{axis_name}_{axis}"] = [mean_lpips]
        dic_results[f"ssim mmse_{axis_name}_{axis}"] = [mean_ssim]
        vec_lpips_mmse.append(mean_lpips)
        vec_psnr_mmse.append(mean_psnr)
        vec_ssim_mmse.append(mean_ssim)
    
    # save data in a csv file 
    os.makedirs(save_metric_path, exist_ok=True)
    fid_pd = pd.DataFrame(dic_results)
    save_dir_metrics = os.path.join(save_metric_path, f'fid_psnr_lpips_ssim_results_eta_{eta}_ddim.csv') #_eta_{eta}_zeta_{zeta}
    fid_pd.to_csv(save_dir_metrics, mode='a', header=not os.path.exists(save_dir_metrics))
    
    # plot the results
  
    plot = plt.figure()
    plt.plot(np.array(axis_vec), np.array(vec_fid_mmse), "b.", label = "FID")
    plt.xlabel(axis_name)
    plt.ylabel("FID")
    plt.title(f"{val_name} versus FID")
    plt.legend()
    plt.savefig(os.path.join(save_metric_path, f"fid_results_eta_{eta}_{val_name}_{val}.png"))
    plt.close("all")
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(axis_vec), np.array(vec_lpips_mmse), "b.", label = "lpips")
    plt.legend()
    plt.xlabel(val_name)
    plt.ylabel("LPIPs")
    plt.title(f"{val_name} versus lpips")
    plt.savefig(os.path.join(save_metric_path, f"lpips_results_{val_name}_{val}.png")) # _eta_{eta}_zeta{zeta}
    plt.close("all")
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(axis_vec), np.array(vec_psnr_mmse), "b.", label = "psnr")
    plt.legend()
    plt.xlabel(val_name)
    plt.ylabel("PSNR")
    plt.title(f"{val_name} versus psnr")
    plt.savefig(os.path.join(save_metric_path, f"psnr_results_{val_name}_{val}.png")) # _eta_{eta}_zeta{zeta}
    plt.close("all")
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(axis_vec), np.array(vec_ssim_mmse), "b.", label = "ssim")
    plt.legend()
    plt.xlabel(val_name)
    plt.ylabel("SSIM")
    plt.title(f"{val_name} versus ssim")
    plt.savefig(os.path.join(save_metric_path, f"ssim_results_{val_name}_{val}.png")) #_eta_{eta}_zeta{zeta}
    plt.close("all")
    
if __name__ == "__main__":
    eta = 0.0
    T_AUG = 4
    solver = "ddpm"
    max_iter = 50
    ckpt_epoch = 600
    date_eval = "22-10-2024"
    ddpm_params = [1] if solver == "ddpm" else [1]
    kargs = DotDict(
            {
              "NUM_TIMESTEPS":TIMESTEPS,  
              "ZETA":ZETA,
              "ddpm_params":ddpm_params,
              "eta":eta,
              "T_AUG":T_AUG,
              "date_eval":date_eval,
              "epoch":ckpt_epoch,
              "max_iter":max_iter,
              "task":"debur",
              "max_unfolded_iter":3
            }
        )
    
    main_metrics(kargs, solver, axis_vec=ZETA,axis_name="Zeta", val=TIMESTEPS[0], val_name="T")