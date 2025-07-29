#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/cmk2000/cmk2000/Deep learning models/LLMs/Conditional-Diffusion-Models-for-IVP/Adversarial-Conditional-Diffusion-Models"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_train_dif.output
#SBATCH -e e_train_dif.error
## Job name
#SBATCH -J CDM_DEBLUR
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=150:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=10000
## GPU requirements
#SBATCH --gres gpu:1
#SBATCH --qos=prilimits
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

# paremeters
###parameters
dataset_name="LSUN" # "ImageNet", "LSUN", "FFHQ"
task="deblur"
sigma_model=0.025
op="gaussian"

#===========================
#  Create results directory
#---------------------------
RESULTS_DIR1="/users/cmk2000/cmk2000/Deep learning models/LLMs/Z_logs/Logger_${SLURM_JOB_NAME}_${dataset_name}_${task}"
RESULTS_DIR=$RESULTS_DIR1"/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR1"

#===============================
#  Application launch commands
#-------------------------------
# Running python scripts

if [ "$dataset_name" = "ImageNet" ]; then
    config_file="imagenet_configs.yaml"
elif [ "$dataset_name" = "LSUN" ]; then
    config_file="fine_tuning_lmodel.yaml"
elif [ "$dataset_name" = "FFHQ" ]; then
    config_file="fine_tuning_lmodel.yaml"
else
    echo "Unsupported dataset: $dataset_name"
    exit 1
fi

python3 main.py \
                --task "$task" \
                --operator_name "$op" \
                --lambda_ 1.0 \
                --config_file "$config_file" \
                --dataset_name "$dataset_name" \
                --sigma_model "$sigma_model" \
                --max_unfolded_iter 3 \
                --kernel_size 25 \
            > "$RESULTS_DIR"_out_train_dif_gpu.output \
            2> "$RESULTS_DIR"_err_train_dif_gpu.error

# Final message
echo "Finish!!"