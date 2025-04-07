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
#SBATCH -J CDM-Inference
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=140:00:00
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
RESULTS_DIR1="/users/cmk2000/cmk2000/Deep learning models/LLMs/Z_logs/Logger_${SLURM_JOB_NAME}"
RESULTS_DIR=$RESULTS_DIR1"/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR1"

#===============================
#  Application launch commands
#-------------------------------
# Running python scripts
python3 main_sample_conditional.py --task "deblur" --operator_name "gaussian" --lambda_ 0.5 --ckpt_epoch 129  --sigma_model 0.05  --ckpt_date 11-03-2025 --config_file "fine_tuning_lmodel.yaml" --total_steps 5 > "$RESULTS_DIR"_out_sample_dif.output 2> "$RESULTS_DIR"_err_sample_dif.error

# Final message
echo "Finish!!"
