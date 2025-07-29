#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "working directory path"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_sample_dif.output
#SBATCH -e e_sample_dif.error
## Job name
#SBATCH -J Inf_JPEG
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=10:00:00
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

fid_eval=0
sigma=0.01
dataset_name="ImageNet"
ckpt_dir=path/to/ckpt
ckpt_name=checkpoint_name
task="jpeg"
for lamb in 1 ; do
    if [ $fid_eval -eq 0 ]; then
        echo "Running CDM with lambda: $lamb and sigma: $sigma"
        python3 sample.py   --task $task \
                            --operator_name "gaussian" \
                            --lambda_ $lamb \
                            --sigma_model $sigma  \
                            --config_file "imagenet_configs_jpeg.yaml" \
                            --dataset $dataset_name \
                            --ckpt_dir $ckpt_dir \
                            --ckpt_name $ckpt_name \
            > "$RESULTS_DIR"_out_sample_dif.output \
            2> "$RESULTS_DIR"_err_sample_dif.error
        
    elif [ $fid_eval = 1 ]; then
        echo "Computing FID: lambda: $lamb and sigma: $sigma"
        python3 fid.py \
                    -d path/to/generated/data/and/ground/truth \
                > "$RESULTS_DIR"_out_fid.output \
                2> "$RESULTS_DIR"_err_fid.error
    else
        echo "The script did not run properly. Please check the parameters. We have two options: fid_eval=0 or fid_eval=1"
        echo "Option 1: fid_eval=0 for sampling."
        echo "Option 2: fid_eval=1 for FID evaluation."
    fi
done

# Final message
echo "Finish!!"
