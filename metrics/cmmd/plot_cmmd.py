import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import os

date = "22-10-2024"
T_AUG = 4
epoch = 300
max_iter =1
timesteps = 100
task = "debur"
max_unfolded_iter = 3
cmmds = []
# ZETA = [0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
ZETA = [0.0]
TIMESTEPS = [20, 50, 100, 150, 200, 250, 350, 500, 600, 700, 800, 900, 1000]
for zeta in ZETA:
    for timesteps in TIMESTEPS:
        dir_=f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/22-10-2024/Max_iter_1/unroll_iter_3/timesteps_{timesteps}/Epoch_{epoch}/T_AUG_4/zeta_{zeta}/eta_0.0"
        dir_file = os.path.join(dir_, f"cmmd_results_zeta_{zeta}_eta_0.0_timesteps_{timesteps}.csv")
        data = pd.read_csv(dir_file)
        cmmd_val = data["cmmd"][0]
        cmmds.append(cmmd_val)
    
data_s = pd.DataFrame({
    "T":TIMESTEPS,
    "cmmd":cmmds
})
save_metric_path = os.path.join(f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/{task}/Results/{date}/Max_iter_{max_iter}/unroll_iter_{max_unfolded_iter}", "metrics_results_calib_timesteps/cmmd_results_T_AUG_{T_AUG}_eta_0.0_zeta_{zeta}.csv")
data_s.to_csv(save_metric_path, mode='a', header=not os.path.exists(save_metric_path))

# plot 
plot = plt.figure()
plt.plot(TIMESTEPS, cmmds, "b.", label = "cmmd")
plt.ylabel("cmmd")
plt.xlabel("T")
plt.legend()
plt.title("cmmd & T")
plt.savefig(os.path.join(f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/22-10-2024/Max_iter_1/unroll_iter_3/metrics_results_calib_timesteps", f"cmmd_results_T_AUG_{T_AUG}_eta_0.0.png"))

# /users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/debur/Results/18-10-2024/Max_iter_1/unroll_iter_3/timesteps_100/Epoch_210/T_AUG_4/zeta_0.0/eta_0.0/cmmd_results_zeta_0.0_eta_0.0.csv