import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metrics.fid_batch.fid_evaluation import Fid_evatuation
# from metrics.cmmd.cmmd import compute_cmmd, main_cmmd

TIMESTEPS = [50, 100, 300, 600, 700, 1000]

def eval(root_dir, save_dir, device):
    dic_results = {}
    save_metric_path = os.path.join(save_dir, "metrics_results_calib_timesteps")
    vec_fid_ddim = []
    vec_fid_ddpm = []
    vec_fid_ours = []
    for t in TIMESTEPS:
        # 
        # ddim
        fid_ddim = Fid_evatuation(root_dir["ddim"](t), device, method="ddim")
        vec_fid_ddim.append(fid_ddim)
       
        #ddpm   
        fid_ddpm = Fid_evatuation(root_dir["ddpm"](t), device, method="ddpm")
        vec_fid_ddpm.append(fid_ddpm)

        # ours
        if t == 300:
            t_ = 350
        else:
            t_ = t
        cmmd_ours = Fid_evatuation(root_dir["ours"](t_), device)
        vec_fid_ours.append(cmmd_ours)
        
        # cmmd
        # # ddim
        # cmmd_ddim = main_cmmd(os.path.join(root_dir["ddim"](t), "last"),os.path.join(root_dir["ddim"](t), "ref"), f"ddim_{t}")
        # vec_cmmd_ddim.append(cmmd_ddim)
       
        # #ddpm   
        # cmmd_ddpm = main_cmmd(os.path.join(root_dir["ddpm"](t), "last"),os.path.join(root_dir["ddpm"](t), "ref"), f"ddpm_{t}")
        # vec_cmmd_ddpm.append(cmmd_ddpm)

        # # ours
        # cmmd_ours= main_cmmd(os.path.join(root_dir["ours"](t), "mmse"),os.path.join(root_dir["ours"](t), "ref"), f"ours_{t}")
        # vec_cmmd_ours.append(cmmd_ours)
   
    dic_results[f"fid_ddim"] = vec_fid_ddim
    dic_results[f"fid_ddpm"] = vec_fid_ddpm
    dic_results[f"fid_ours"] = vec_fid_ours
    
   
    dic_results[f"timesteps"] = TIMESTEPS
    
    # save data in a csv file 
    os.makedirs(save_metric_path, exist_ok=True)
    fid_pd = pd.DataFrame(dic_results)
    save_dir_fid_cmmd = os.path.join(save_metric_path, f'fid_cmmd_metrics.csv') #_eta_{eta}_zeta{zeta}
    fid_pd.to_csv(save_dir_fid_cmmd, mode='a', header=not os.path.exists(save_dir_fid_cmmd))
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(TIMESTEPS), np.array(vec_fid_ddim), "b.", label = "DDIM")
    plt.plot(np.array(TIMESTEPS), np.array(vec_fid_ddpm), "b.", label = "DDPM")
    plt.plot(np.array(TIMESTEPS), np.array(vec_fid_ours), "b.", label = "OURS")
    plt.legend()
    plt.xlabel("T")
    plt.ylabel("FID")
    plt.title("T versus FID")
    plt.savefig(os.path.join(save_metric_path, f"fid_score.png")) # _eta_{eta}_zeta{zeta}
    plt.close("all")
    
    
 
if __name__ == "__main__":
    ddim_ph = lambda t: f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/07-11-2024/ddim/no_lora/Simple sampling/timesteps_{t}/zeta_0.2/eta_0.0"
    ddpm_ph = lambda t: f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/07-11-2024/ddpm/no_lora/Simple sampling/timesteps_{t}/ddpm_param_3"
    ours_ph = lambda t: f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/29-10-2024/ddpm/Max_iter_1/unroll_iter_3/timesteps_{t}/Epoch_600/T_AUG_4/ddpm_param_3"
    
    root_dir = {
        "ddim":ddim_ph,
        "ddpm":ddpm_ph,
        "ours":ours_ph
    }
    save_dir = "/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval(root_dir, save_dir, device)