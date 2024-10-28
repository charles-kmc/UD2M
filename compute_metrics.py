import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from metrics.fid_batch.fid_evaluation import Fid_evatuation

# ZETA = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
ZETA = [0.0]
TIMESTEPS = [20, 50, 100, 150, 200, 250, 350, 500, 600, 700, 800, 900, 1000]
def main(eta, T_AUG):
    date = "22-10-2024"
    epoch = 300
    max_iter =1
    # timesteps = 100
    task = "debur"
    max_unfolded_iter = 3
    
    dic_results = {}
    vec_psnr_mmse = []
    vec_ssim_mmse = []
    vec_lpips_mmse = []
    for zeta in ZETA:
        for timesteps in TIMESTEPS:
            root_dir_csv = f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/{task}/Results/{date}/Max_iter_{max_iter}/unroll_iter_{max_unfolded_iter}/timesteps_{timesteps}/Epoch_{epoch}/T_AUG_{T_AUG}/zeta_{zeta}/eta_{eta}/metrics_results"
            save_metric_path = os.path.join(f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/{task}/Results/{date}/Max_iter_{max_iter}/unroll_iter_{max_unfolded_iter}", "metrics_results_calib_timesteps")
            data = pd.read_csv(os.path.join(root_dir_csv, "swd_results.csv"))
            mean_psnr = data["psnr mean"].mean()
            mean_lpips = data["lpips mean"].mean()
            mean_ssim = data["ssim mean"].mean()
            dic_results[f"psnr mmse_timesteps{timesteps}"] = [mean_psnr]
            dic_results[f"lpips mmse_timesteps{timesteps}"] = [mean_lpips]
            dic_results[f"ssim mmse_timesteps{timesteps}"] = [mean_ssim]
            vec_lpips_mmse.append(mean_lpips)
            vec_psnr_mmse.append(mean_psnr)
            vec_ssim_mmse.append(mean_ssim)
    
    # save data in a csv file 
    os.makedirs(save_metric_path, exist_ok=True)
    fid_pd = pd.DataFrame(dic_results)
    save_dir_ssim_psnr_lpips = os.path.join(save_metric_path, f'metrics_psnr_ssim_lpips_results_T_AUG_{T_AUG}_eta_{eta}_zeta{zeta}.csv')
    fid_pd.to_csv(save_dir_ssim_psnr_lpips, mode='a', header=not os.path.exists(save_dir_ssim_psnr_lpips))
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(TIMESTEPS), np.array(vec_lpips_mmse), "b.", label = "lpips")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("LPIPs")
    plt.title("T versus lpips")
    plt.savefig(os.path.join(save_metric_path, f"lpips_results_T_AUG_{T_AUG}_eta_{eta}_zeta{zeta}.png"))
    plt.close("all")
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(TIMESTEPS), np.array(vec_psnr_mmse), "b.", label = "psnr")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("PSNR")
    plt.title("zeta versus psnr")
    plt.savefig(os.path.join(save_metric_path, f"psnr_results_T_AUG_{T_AUG}_eta_{eta}_zeta{zeta}.png"))
    plt.close("all")
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(TIMESTEPS), np.array(vec_ssim_mmse), "b.", label = "ssim")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("SSIM")
    plt.title("T versus ssim")
    plt.savefig(os.path.join(save_metric_path, f"ssim_results_T_AUG_{T_AUG}_eta_{eta}_zeta{zeta}.png"))
    plt.close("all")
    
if __name__ == "__main__":
    eta = 0.0
    T_AUG = 4
    main(eta, T_AUG)