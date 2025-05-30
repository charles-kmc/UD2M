import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metrics.fid_batch.fid_evaluation import Fid_evatuation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ZETA = [0.6]
TIMESTEPS = [3]
def main(cfg):
    T_AUG = cfg.T_AUG
    timesteps  = cfg.timesteps
    dic_results = {}
    vec_psnr_mmse = []
    vec_ssim_mmse = []
    vec_lpips_mmse = []
    dic_results[f"T"] = [timesteps]
    root_dir = cfg.root_dir
    save_metric_path = cfg.root_dir
    
    # FID
    out = Fid_evatuation(root_dir, device, mmse_sample=True, last_sample=True)
    dic_results[f"fid mmse"] = [out["fid_mmse"]]
    dic_results[f"fid last"] = [out["fid_last"]]
    dic_results[f"fid init"] = [out["fid_init"]]
    
    data = pd.read_csv(os.path.join(root_dir, "metrics_results","metrics_results.csv"))
    mean_psnr = data["psnr"].mean()
    mean_lpips = data["lpips"].mean()
    mean_ssim = data["ssim"].mean()
    mean_psnr_l = data["psnr last"].mean()
    mean_lpips_l = data["lpips last"].mean()
    mean_ssim_l = data["ssim last"].mean()
    mean_psnr_i = data["psnr_init"].mean()
    mean_lpips_i = data["lpips_init"].mean()
    dic_results[f"psnr mmse"] = [mean_psnr]
    dic_results[f"lpips mmse"] = [mean_lpips]
    dic_results[f"ssim mmse"] = [mean_ssim]
    dic_results[f"psnr sample"] = [mean_psnr_l]
    dic_results[f"lpips sample"] = [mean_lpips_l]
    dic_results[f"ssim sample"] = [mean_ssim_l]
    dic_results[f"psnr init"] = [mean_psnr_i]
    dic_results[f"lpips init"] = [mean_lpips_i]
    vec_lpips_mmse.append(mean_lpips)
    vec_psnr_mmse.append(mean_psnr)
    vec_ssim_mmse.append(mean_ssim)

    # save data in a csv file 
    os.makedirs(save_metric_path, exist_ok=True)
    fid_pd = pd.DataFrame(dic_results)
    save_dir_ssim_psnr_lpips = os.path.join(save_metric_path, f'metrics_psnr_ssim_lpips_results.csv') #_eta_{eta}_zeta{zeta}
    fid_pd.to_csv(save_dir_ssim_psnr_lpips, mode='w', header=True)
    
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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3, help="number of timesteps")
    parser.add_argument("--T_AUG", type=int, default=5, help="number of T_AUG")
    parser.add_argument('--root_dir', required=True, type=str, help='root directory for the results')
    args = parser.parse_args()
    main(args)