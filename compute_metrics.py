import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metrics.fid_batch.fid_evaluation import Fid_evatuation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ZETA = [0.6]
TIMESTEPS = [100, 500, 1000]
def main(T_AUG):
    dic_results = {}
    vec_psnr_mmse = []
    vec_ssim_mmse = []
    vec_lpips_mmse = []
    for timesteps in TIMESTEPS:
        dic_results[f"T"] = [timesteps]
        root_dir = f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/inp/Results_sampling_decreasing_lambda_free_noise_model/18-12-2024/ddim/lora/Max_iter_1/unroll_iter_3/timesteps_{timesteps}/Epoch_300/T_AUG_0/zeta_0.6/eta_0.0/stochastic_0.05/lambda_0.003"
        save_metric_path = f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/inp/Results_sampling_decreasing_lambda_free_noise_model/18-12-2024/ddim/lora/Max_iter_1/unroll_iter_3"
        
        # FID
        out = Fid_evatuation(root_dir, device, mmse_sample=True, last_sample=True)
        dic_results[f"fid mmse"] = [out["fid_mmse"]]
        dic_results[f"fid last"] = [out["fid_last"]]
        
        data = pd.read_csv(os.path.join(root_dir, "metrics_results","metrics_results.csv"))
        mean_psnr = data["psnr"].mean()
        mean_lpips = data["lpips"].mean()
        mean_ssim = data["ssim"].mean()
        mean_psnr_l = data["psnr last"].mean()
        mean_lpips_l = data["lpips last"].mean()
        mean_ssim_l = data["ssim last"].mean()
        dic_results[f"psnr mmse"] = [mean_psnr]
        dic_results[f"lpips mmse"] = [mean_lpips]
        dic_results[f"ssim mmse"] = [mean_ssim]
        dic_results[f"psnr sample"] = [mean_psnr_l]
        dic_results[f"lpips sample"] = [mean_lpips_l]
        dic_results[f"ssim sample"] = [mean_ssim_l]
        vec_lpips_mmse.append(mean_lpips)
        vec_psnr_mmse.append(mean_psnr)
        vec_ssim_mmse.append(mean_ssim)
    
        # save data in a csv file 
        os.makedirs(save_metric_path, exist_ok=True)
        fid_pd = pd.DataFrame(dic_results)
        save_dir_ssim_psnr_lpips = os.path.join(save_metric_path, f'metrics_psnr_ssim_lpips_results.csv') #_eta_{eta}_zeta{zeta}
        fid_pd.to_csv(save_dir_ssim_psnr_lpips, mode='a', header=not os.path.exists(save_dir_ssim_psnr_lpips))
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(TIMESTEPS), np.array(vec_lpips_mmse), "b.", label = "lpips")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("LPIPs")
    plt.title("T versus lpips")
    plt.savefig(os.path.join(save_metric_path, f"lpips_results_T_AUG_{T_AUG}.png")) # _eta_{eta}_zeta{zeta}
    plt.close("all")
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(TIMESTEPS), np.array(vec_psnr_mmse), "b.", label = "psnr")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("PSNR")
    plt.title("zeta versus psnr")
    plt.savefig(os.path.join(save_metric_path, f"psnr_results_T_AUG_{T_AUG}.png")) # _eta_{eta}_zeta{zeta}
    plt.close("all")
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(TIMESTEPS), np.array(vec_ssim_mmse), "b.", label = "ssim")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("SSIM")
    plt.title("T versus ssim")
    plt.savefig(os.path.join(save_metric_path, f"ssim_results_T_AUG_{T_AUG}.png")) #_eta_{eta}_zeta{zeta}
    plt.close("all")
    
if __name__ == "__main__":
    T_AUG = 5
    main(T_AUG)