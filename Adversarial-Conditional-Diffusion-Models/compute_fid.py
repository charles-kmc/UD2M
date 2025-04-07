import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import DotDict

from metrics.fid_batch.fid_evaluation import Fid_evatuation


class Evaluation:
    def __init__(self, TIMESTEPS, args):
        self.TIMESTEPS=TIMESTEPS
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def HUEN(self, lambda_:float, solver:str="huen"):

        dic_results = {}
        vec_fid_mmse = []
        vec_psnr_mmse = []
        vec_ssim_mmse = []
        vec_lpips_mmse = []
    
        for timesteps in self.TIMESTEPS:
            huen_resutls = f"/users/cmk2000/sharedscratch/CKM/Regularised_CDM/Results/{self.args.task}_lora/operator_{self.args.operator}/{self.args.date}/{solver}/Max_iter_1/timesteps_{timesteps}/{solver}/lambda_{lambda_}"
            
            save_metric_path = os.path.join(huen_resutls, "summary")
            os.makedirs(save_metric_path, exist_ok=True)
            
            # FID
            out = Fid_evatuation(huen_resutls, self.device, mmse_sample=True, last_sample=self.args.sample_size>1)
            vec_fid_mmse.append(out["fid_mmse"])
            dic_results[f"fid mmse_vec_{timesteps}"] = [out["fid_mmse"]]
            dic_results[f"fid last_vec_{timesteps}"] = [out["fid_last"]]
            
            print(f"fid_last = {out['fid_last']}")
            print(f"fid_mmse = {out['fid_mmse']}")
        
            # other metrics: PSNR, SSIM, LPIPs
            root_dir_csv = os.path.join(huen_resutls, "metrics_results")
            data = pd.read_csv(os.path.join(root_dir_csv, "metrics_results.csv"))
            mean_psnr = data["psnr"].mean()
            mean_lpips = data["lpips"].mean()
            mean_ssim = data["ssim"].mean()
            dic_results[f"psnr mmse_{timesteps}"] = [mean_psnr]
            dic_results[f"lpips mmse_{timesteps}"] = [mean_lpips]
            dic_results[f"ssim mmse_{timesteps}"] = [mean_ssim]
            vec_lpips_mmse.append(mean_lpips)
            vec_psnr_mmse.append(mean_psnr)
            vec_ssim_mmse.append(mean_ssim)
        
        # save data in a csv file 
        os.makedirs(save_metric_path, exist_ok=True)
        fid_pd = pd.DataFrame(dic_results)
        save_dir_metrics = os.path.join(save_metric_path, f'summary_results_{timesteps}_{solver}.csv') #_eta_{eta}_zeta_{zeta}
        print("save dir:", save_dir_metrics)
        fid_pd.to_csv(save_dir_metrics, mode='a', header=not os.path.exists(save_dir_metrics))
        
        if len(self.TIMESTEPS)>1:
            dicts = {
                "FID":vec_fid_mmse,
                "PSNR":vec_psnr_mmse,
                "LPIP":vec_lpips_mmse,
                "SSIM":vec_ssim_mmse,
            }
            self.plot(dicts, "T", save_metric_path)
            


    def DDPM(self, lambda_:float, ddpm_param:float, solver:str="ddpm"):
        
        dic_results = {}
        vec_fid_mmse = []
        vec_psnr_mmse = []
        vec_ssim_mmse = []
        vec_lpips_mmse = []
    
        for timesteps in self.TIMESTEPS:
            ddpm_resutls = f"/users/cmk2000/sharedscratch/CKM/Regularised_CDM/Results/{self.args.task}_lora/operator_{self.args.operator}/{self.args.date}/{solver}/Max_iter_1/timesteps_{timesteps}/ddpm_param_{ddpm_param}/lambda_{lambda_}"
            
            save_metric_path = os.path.join(ddpm_resutls, "summay")

            # FID
            out = Fid_evatuation(ddpm_resutls, self.device, mmse_sample=True, last_sample=self.args.sample_size>1)
            vec_fid_mmse.append(out["fid_mmse"])
            dic_results[f"fid mmse_vec_{timesteps}"] = [out["fid_mmse"]]
            dic_results[f"fid last_vec_{timesteps}"] = [out["fid_last"]]
            
            print(f"fid_last = {out['fid_last']}")
            print(f"fid_mmse = {out['fid_mmse']}")
        
            # other metrics: PSNR, SSIM, LPIPs
            root_dir_csv = os.path.join(ddpm_resutls, "metrics_results")
            data = pd.read_csv(os.path.join(root_dir_csv, "metrics_results.csv"))
            mean_psnr = data["psnr"].mean()
            mean_lpips = data["lpips"].mean()
            mean_ssim = data["ssim"].mean()
            dic_results[f"psnr mmse_{timesteps}"] = [mean_psnr]
            dic_results[f"lpips mmse_{timesteps}"] = [mean_lpips]
            dic_results[f"ssim mmse_{timesteps}"] = [mean_ssim]
            vec_lpips_mmse.append(mean_lpips)
            vec_psnr_mmse.append(mean_psnr)
            vec_ssim_mmse.append(mean_ssim)
        
        # save data in a csv file 
        os.makedirs(save_metric_path, exist_ok=True)
        fid_pd = pd.DataFrame(dic_results)
        save_dir_metrics = os.path.join(save_metric_path, f'summary_results_{timesteps}_{solver}.csv') #_eta_{eta}_zeta_{zeta}
        fid_pd.to_csv(save_dir_metrics, mode='a', header=not os.path.exists(save_dir_metrics))
        
        if len(self.TIMESTEPS)>1:
            dicts = {
                "FID":vec_fid_mmse,
                "PSNR":vec_psnr_mmse,
                "LPIP":vec_lpips_mmse,
                "SSIM":vec_ssim_mmse,
            }
            self.plot(dicts, "T", save_metric_path)
            
    

    def DDIM(self, lambda_:float, eta:float, zeta:float, solver:str="ddim"):
        
        dic_results = {}
        vec_fid_mmse = []
        vec_psnr_mmse = []
        vec_ssim_mmse = []
        vec_lpips_mmse = []
    
        for timesteps in self.TIMESTEPS:
            ddim_resutls = f"/users/cmk2000/sharedscratch/CKM/Regularised_CDM/Results/{self.args.task}_lora/operator_{self.args.operator}/{self.args.date}/{solver}/Max_iter_1/timesteps_{timesteps}/zeta_{zeta}/eta_{eta}/lambda_{lambda_}"
            
            save_metric_path = os.path.join(ddim_resutls, "summay")
            os.makedirs(save_metric_path, exist_ok=True)
            # FID
            out = Fid_evatuation(ddim_resutls, self.device, mmse_sample=True, last_sample=self.args.sample_size>1)
            vec_fid_mmse.append(out["fid_mmse"])
            dic_results[f"fid mmse_vec_{timesteps}"] = [out["fid_mmse"]]
            dic_results[f"fid last_vec_{timesteps}"] = [out["fid_last"]]

            print(f"fid_last = {out['fid_last']}")
            print(f"fid_mmse = {out['fid_mmse']}")
        
            # other metrics: PSNR, SSIM, LPIPs
            root_dir_csv = os.path.join(ddim_resutls, "metrics_results")
            data = pd.read_csv(os.path.join(root_dir_csv, "metrics_results.csv"))
            mean_psnr = data["psnr"].mean()
            mean_lpips = data["lpips"].mean()
            mean_ssim = data["ssim"].mean()
            dic_results[f"psnr mmse_{timesteps}"] = [mean_psnr]
            dic_results[f"lpips mmse_{timesteps}"] = [mean_lpips]
            dic_results[f"ssim mmse_{timesteps}"] = [mean_ssim]
            vec_lpips_mmse.append(mean_lpips)
            vec_psnr_mmse.append(mean_psnr)
            vec_ssim_mmse.append(mean_ssim)
        
        # save data in a csv file 
        os.makedirs(save_metric_path, exist_ok=True)
        fid_pd = pd.DataFrame(dic_results)
        save_dir_metrics = os.path.join(save_metric_path, f'summary_results_{timesteps}_{solver}.csv') #_eta_{eta}_zeta_{zeta}
        fid_pd.to_csv(save_dir_metrics, mode='a', header=not os.path.exists(save_dir_metrics))
        
        if len(self.TIMESTEPS)>1:
            dicts = {
                "FID":vec_fid_mmse,
                "PSNR":vec_psnr_mmse,
                "LPIP":vec_lpips_mmse,
                "SSIM":vec_ssim_mmse,
            }
            self.plot(dicts, "T", save_metric_path)
        
    def plot(self, dicts, label, save_metric_path):
        for key, value in dicts.items():
            # plot the results
            plot = plt.figure()
            plt.plot(np.array(TIMESTEPS), np.array(value), "b.", label = key)
            plt.xlabel(label)
            plt.ylabel("FID")
            plt.title(f"{label} versus FID")
            plt.legend()
            plt.savefig(os.path.join(save_metric_path, f"{key}_results_{label}.png"))
            plt.close("all")
        

if __name__ == "__main__":
    
    
    # task = "deblur"    # inp, sr, ct
    # date = "01-03-2025"
    # operator= "uniform_motion" #"box"
    # lambda_=0.19
    
    task = "deblur"    # inp, sr, ct
   
    
    if task=="inp":
        date = "25-03-2025"
        operator= "random" #"box"
        lambda_=0.0005
        TIMESTEPS = [50]
        #lambda_=0.001
        kargs = DotDict({
                    "date": date,
                    "task":task,
                    "operator":operator,
                    "sample_size":1,
                    })
    elif task=="sr":
        date = "01-03-2025"
        operator= "gaussian"
        lambda_=10.
        TIMESTEPS = [1]
        #lambda_=10.0
        kargs = DotDict({
                    "date": date,
                    "task":task,
                    "operator":operator,
                    "sample_size":1,
                    })
    elif task=="deblur":
        date = "26-03-2025"#"01-03-2025"
        operator= "gaussian"#"uniform_motion" 
        lambda_= 0.2#0.19
        TIMESTEPS = [3]#[2]
        kargs = DotDict({
                    "date": date,
                    "task":task,
                    "operator":operator,
                    "sample_size":1,
                    })
    else:
        raise ValueError("task not implemented!!")
    
    
    eval = Evaluation(TIMESTEPS, kargs)
    
    # eval.HUEN(lambda_)
    
    zeta=0.1
    eta = 0.0
    eval.DDIM(lambda_, eta, zeta)
    
    # ddpm_param=1.
    # eval.DDPM(lambda_,ddpm_param) 