#!/bin/bash

# Define the root directory
date="22-10-2024"
epoch=300
max_iter=1
timesteps=100
task="debur"
max_unfolded_iter=3
eta=0.0
T_AUG=4
zeta=0.0
# for zeta in 0.1 0.2 0.3 0.4 0.5 0.6 1.0; do
for timesteps in 20 50 100 150 200 250 350 500 600 700 800 900 1000; do
    dir_ours="/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/${task}/Results/${date}/Max_iter_${max_iter}/unroll_iter_${max_unfolded_iter}/timesteps_${timesteps}/Epoch_${epoch}/T_AUG_${T_AUG}/zeta_${zeta}/eta_${eta}"
    dir_ours_ref="${dir_ours}/ref"
    output_file="timesteps_${timesteps}_deb.out"
    dir_ours_mmse="${dir_ours}/mmse"
    if [ -d "$dir_ours_ref" ]; then
        echo "Directory exists."
    else
        echo "Directory does not exist."
    fi
    echo "$dir_ours_mmse"
    python3 -m main_deb "${dir_ours_ref}" "${dir_ours_mmse}" "${eta}" "${zeta}" "${timesteps}" >> "${output_file}"
done