#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "working directory path"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_train_dif.output
#SBATCH -e e_train_dif.error
## Job name
#SBATCH -J CDM
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=140:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=20000
## GPU requirements
#SBATCH --gres gpu:1
##SBATCH --qos=prilimits
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

###parameters
dataset_name="LSUN"
task="inp"
sigma_model=0.025
operator_name="box"
#===========================
#  Create results directory
#---------------------------
RESULTS_DIR1=""
RESULTS_DIR=$RESULTS_DIR1"/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR1"

#===============================
#  Application launch commands
#-------------------------------
# Running python scripts

if [ "$dataset_name" = "LSUN" ]; then
    config_file="fine_tuning_lmodel.yaml"
elif [ "$dataset_name" = "ImageNet" ]; then
    config_file="imagenet_configs.yaml"
else
    echo "Dataset not recognized. Exiting."
    exit 1
fi
python3 main.py \
            --task $task \
            --operator_name $operator_name \
            --lambda_ 1 \
            --config_file $config_file \
            --dataset_name $dataset_name \
            --max_unfolded_iter 3 \
            --learned_lambda_sig True \
            --use_RAM True \
            --sigma_model $sigma_model \
        > "$RESULTS_DIR"_out_train_dif.output \
        2> "$RESULTS_DIR"_err_train_dif.error 
echo "Finish!!"

