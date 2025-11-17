import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metrics.fid_batch.fid_evaluation import Fid_evatuation_numpy


def run_fid(directory:str, last_sample:bool):
    save_metric_path = os.path.join(directory, "summary")
    os.makedirs(save_metric_path, exist_ok=True)
    dic_results = {}   

    # FID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        out = Fid_evatuation_numpy(directory, device, mmse_sample=True, last_sample=last_sample)
    dic_results[f"fid mmse_vec"] = [out["fid_mmse"].item()]
    dic_results[f"fid last_vec"] = [out["fid_last"].item()]
    dic_results[f"cfid mmse_vec"] = [out["cfid_mmse"].item()]
    dic_results[f"cfid last_vec"] = [out["cfid_last"].item()]

    print(f"fid_last = {out['fid_last']}")
    print(f"fid_mmse = {out['fid_mmse']}")
    print(f"cfid_last = {out['cfid_last']}")
    print(f"cfid_mmse = {out['cfid_mmse']}")

    # other metrics: PSNR, SSIM, LPIPs
    try:
        root_dir_csv = os.path.join(directory, "metrics_results")
        data = pd.read_csv(os.path.join(root_dir_csv, "metrics_results.csv"))
        data_new = data
        mean_psnr = data_new["psnr"].mean()
        mean_lpips = data_new["lpips"].mean()
        mean_ssim = data_new["ssim"].mean()
        mean_psnr_last = data_new["psnr last"].mean()
        mean_lpips_last = data_new["lpips last"].mean()
        mean_ssim_last = data_new["ssim last"].mean()
        dic_results[f"psnr mmse"] = [mean_psnr]
        dic_results[f"psnr last"] = [mean_psnr_last]
        dic_results[f"lpips mmse"] = [mean_lpips]
        dic_results[f"lpips last"] = [mean_lpips_last]
        dic_results[f"ssim mmse"] = [mean_ssim]
        dic_results[f"ssim last"] = [mean_ssim_last]
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        dic_results[f"psnr mmse"] = [0]
        dic_results[f"lpips mmse"] = [0]
        dic_results[f"ssim mmse"] = [0]
    
    # save data in a csv file 
    os.makedirs(save_metric_path, exist_ok=True)
    fid_pd = pd.DataFrame(dic_results)
    save_dir_metrics = os.path.join(save_metric_path, f'summary_results.csv') 
    fid_pd.to_csv(save_dir_metrics, mode='w', header=True)#not os.path.exists(save_dir_metrics))
    
if __name__ == "__main__":
    # arguments
    import argparse
    parser = argparse.ArgumentParser(description="FID evaluation")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Directory to evaluate")
    parser.add_argument("-l", "--last_sample", type=bool, default=True, help="fid for the last sample")
    args = parser.parse_args()
    
    run_fid(directory = args.directory, last_sample = args.last_sample)