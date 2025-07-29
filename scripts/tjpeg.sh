#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/cmk2000/cmk2000/Deep learning models/LLMs/Regularised_CDM"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_train_dif.output
#SBATCH -e e_train_dif.error
## Job name
#SBATCH -J CDjpeg
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=150:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=10000
## GPU requirements
#SBATCH --qos=prilimits
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

# get number of nodes specified above

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
dataset_name="ImageNet"
task="jpeg"

if [ "$dataset_name" = "LSUN" ]; then
echo "Start training our method on $dataset_name with noise variance 0.01^2 ..."
    python3 train.py \
                    --task $task \
                    --operator_name "gaussian" \
                    --lambda_ 0.4 \
                    --config_file "fine_tuning_lmodel.yaml" \
                    --dataset $dataset_name \
                    --sigma_model 0.01 \
                > "$RESULTS_DIR"_out_train_dif_"$task".output \
                2> "$RESULTS_DIR"_err_train_dif_"$task".error

elif [ "$dataset_name" = "ImageNet" ]; then
    echo "Start training our method on $dataset_name with noise variance 0.01^2 ..."
    python3 train.py \
                    --task $task \
                    --operator_name "gaussian" \
                    --lambda_ 1 \
                    --config_file "imagenet_configs_jpeg.yaml" \
                    --dataset $dataset_name \
                    --sigma_model 0.01 \
                > "$RESULTS_DIR"_out_train_dif_"$task".output \
                2> "$RESULTS_DIR"_err_train_dif_"$task".error
else
    echo "Not yet implemented for $dataset_name !!"
fi

# Final message
echo "Finish!!"