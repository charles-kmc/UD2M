import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from metrics.fid_batch.fid_evaluation import Fid_evatuation

ZETA = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
def main(eta, T_AUG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date = "18-10-2024"
    epoch = 210
    max_iter =1
    timesteps = 100
    task = "debur"
    max_unfolded_iter = 3
    
    dic_results = {}
    vec_fid_mmse = []
    for zeta in ZETA:
        root_dir = f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/{task}/Results/{date}/Max_iter_{max_iter}/unroll_iter_{max_unfolded_iter}/timesteps_{timesteps}/Epoch_{epoch}/T_AUG_{T_AUG}/zeta_{zeta}/eta_{eta}"
        save_metric_path = os.path.join(root_dir, "metrics_results")
        
        fid_mmse, _ = Fid_evatuation(root_dir, device)
        vec_fid_mmse.append(fid_mmse)
        dic_results[f"fid mmse_zeta{zeta}"] = [fid_mmse]
    
    # save data in a csv file 
    fid_pd = pd.DataFrame(dic_results)
    save_dir_fid = os.path.join(save_metric_path, f'fid_results_T_AUG_{T_AUG}_eta_{eta}.csv')
    fid_pd.to_csv(save_dir_fid, mode='a', header=not os.path.exists(save_dir_fid))
    
    # plot the results
    plot = plt.figure()
    plt.plot(np.array(ZETA), np.array(vec_fid_mmse), "b.", label = "FID")
    plt.legend()
    plt.xlabel("zeta")
    plt.ylabel("FID")
    plt.title("zeta versus FID")
    plt.savefig(os.path.join(save_metric_path, f"fid_results_T_AUG_{T_AUG}_eta_{eta}.png"))
    plt.close("all")
    
if __name__ == "__main__":
    eta = 0.0
    T_AUG = 4
    main(eta, T_AUG)