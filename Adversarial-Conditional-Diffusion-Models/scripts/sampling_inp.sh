#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/cmk2000/cmk2000/Deep learning models/LLMs/Conditional-Diffusion-Models-for-IVP/Adversarial-Conditional-Diffusion-Models"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_sample_dif.output
#SBATCH -e e_sample_dif.error
## Job name
#SBATCH -J Inference
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
dataset_name="ImageNet"
fid_eval=0
op="box"
lamb=0.1
sigma=0.01
task="inp"
num_timesteps=5
max_images=5
max_unfolded_iter=3
kernel_size=3

# inp
ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_3_rank_5/imagenet/sigma_model_0.01/inp/box/10-06-2025/scale_0.3"
ckpt_name="imagenet_BestPSNR_lora_inp_box_epoch=142-5.ckpt"


if [ "$dataset_name" = "ImageNet" ]; then
    config_file="imagenet_configs.yaml"
else
    config_file="fine_tuning_lmodel.yaml"
fi

if [ $fid_eval -eq 0 ]; then
    python3 sampling.py   --task $task \
                    --operator_name $op \
                    --lambda_ $lamb \
                    --sigma_model $sigma \
                    --config_file $config_file \
                    --dataset_name $dataset_name \
                    --ckpt_dir "$ckpt_dir" \
                    --ckpt_name "$ckpt_name" \
                    --num_timesteps $num_timesteps \
                    --max_images $max_images \
                    --kernel_size $kernel_size \
                    --max_unfolded_iter $max_unfolded_iter \
        > "$RESULTS_DIR"_out_sample_dif.output \
        2> "$RESULTS_DIR"_err_sample_dif.error

    # python3 fid.py \
    #         -d "/users/cmk2000/sharedscratch/CKM/Regularised_CDM/Results_RAM_unfolded_iter_$max_unfolded_iter/ImageNet/sr_lora/operator_gaussian_3/var_0.01/28-05-2025/ddim/factor_4/timesteps_3/zeta_0.9/eta_0.8/lambda_0.31" \
    #     > "$RESULTS_DIR"_out_fid.output \
    #     2> "$RESULTS_DIR"_err_fid.error
    
elif [ $fid_eval -eq 1 ]; then
    echo "Computing FID: lambda: $lamb and sigma $sigma on dataset_name $dataset_name"          
    python3 fid.py \
            -d "/users/cmk2000/sharedscratch/CKM/Regularised_CDM/Results_RAM_unfolded_iter_$max_unfolded_iter/ImageNet/sr_lora/operator_gaussian_3/var_0.01/28-05-2025/ddim/factor_4/timesteps_3/zeta_0.9/eta_0.8/lambda_0.31" \
        > "$RESULTS_DIR"_out_fid.output \
        2> "$RESULTS_DIR"_err_fid.error
else
    echo "Next..."
fi

# Final message
echo "Finish!!"
