import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cmmd import compute_cmmd, main_cmmd

TIMESTEPS = [200]
ZETA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

def eval(root_dir, save_dir, device, axis_vec = ZETA, val = TIMESTEPS, val_name = "T", axis_name = "Zeta"):
    dic_results = {}
    folder = "last"
    save_metric_path = os.path.join(save_dir, f"metrics_results_calib_timesteps_{folder}")

    vec_cmmd_ours = []
    for axis in axis_vec:
        if axis_name == "Zeta":
            zeta = axis
            t = val[0]
        elif axis_name == "T":
            zeta = val[0]
            t = axis
        else:
            raise NotImplemented("Not Implemented !!")
        # ours
        cmmd_ours= main_cmmd(os.path.join(root_dir["ours"](t, zeta), folder), os.path.join(root_dir["ours"](t,zeta), "ref"), f"ours_T_{t}_zeta_{zeta}")
        vec_cmmd_ours.append(cmmd_ours)
   
    dic_results[f"cmmd_ours"] = vec_cmmd_ours
    dic_results[f"{axis_name}"] = axis_vec
    
    # save data in a csv file 
    os.makedirs(save_metric_path, exist_ok=True)
    fid_pd = pd.DataFrame(dic_results)
    save_dir_fid_cmmd = os.path.join(save_metric_path, f'fid_cmmd_metrics__{val_name}_{val}.csv') #_eta_{eta}_zeta{zeta}
    fid_pd.to_csv(save_dir_fid_cmmd, mode='a', header=not os.path.exists(save_dir_fid_cmmd))
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(axis_vec), np.array(vec_cmmd_ours), "b.", label = "OURS")
    plt.legend()
    plt.xlabel(f"{axis_name}")
    plt.ylabel("CMMD")
    plt.title(f"{axis_name} versus CMMD")
    plt.savefig(os.path.join(save_metric_path, f"cmmd_score_{val_name}_{val}.png")) # _eta_{eta}_zeta{zeta}
    plt.close("all")

if __name__ == "__main__":
    ddim_ph = lambda t, zeta: f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/07-11-2024/ddim/no_lora/Simple sampling/timesteps_{t}/zeta_{zeta}/eta_0.0"
    ddpm_ph = lambda t: f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/07-11-2024/ddpm/no_lora/Simple sampling/timesteps_{t}/ddpm_param_3"
    ours_ph = lambda t,zeta: f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/11-11-2024/ddim/lora/Max_iter_50/unroll_iter_3/timesteps_{t}/Epoch_600/T_AUG_4/zeta_{zeta}/eta_0.0"
    
    root_dir = {
        "ddim":ddim_ph,
        "ddpm":ddpm_ph,
        "ours":ours_ph
    }
    save_dir = "/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/11-11-2024/ddim/lora/Max_iter_50/unroll_iter_3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval(root_dir, save_dir, device)