#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/cmk2000/cmk2000/Deep learning models/LLMs/Regularised_CDM"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_sample_dif.output
#SBATCH -e e_sample_dif.error
## Job name
#SBATCH -J FID
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=1:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=10000
## GPU requirements
#SBATCH --gres gpu:1
##SBATCH --ntasks=32
## Specify partition
#SBATCH -p gpu

################# Part-2 Shell script ####################
#===============================
#  Activate Flight Environment
#-------------------------------
source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

#==============================
#  Activate Package Ecosystem
#------------------------------
# Load the Conda module for access to `conda` command
# flight env activate conda

# Activate the specific Conda environment
conda activate dmog_env39

if [ $? -eq 0 ]; then
    echo "Conda is active and functioning."
else
    echo "Conda did not initialize properly."
    exit 1
fi

#===========================
#  Create results directory
#---------------------------
RESULTS_DIR1="/users/cmk2000/cmk2000/Deep learning models/LLMs/Z_logs/Logger/fid_${SLURM_JOB_NAME}"
RESULTS_DIR=$RESULTS_DIR1"/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR1"

#===============================
#  Application launch commands
#-------------------------------
# Running python scripts
python3 fid.py \
    -d "/users/cmk2000/sharedscratch/CKM/Regularised_CDM/Results_RAM_unfolded_iter_3_rank_5_paper/ImageNet/deblur_lora/operator_motion_5/var_0.01/17-06-2025/ddim/timesteps_3/zeta_0.9/eta_0.8/lambda_4.0" \
    > "$RESULTS_DIR"_out_sample_dif_fid.output \
    2> "$RESULTS_DIR"_err_sample_dif_fid.error

# Final message
echo "Finish!!"
