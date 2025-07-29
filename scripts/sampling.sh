#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/cmk2000/cmk2000/Deep learning models/LLMs/Adversarial-Conditional-Diffusion-Models"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_sample_dif.output
#SBATCH -e e_sample_dif.error
## Job name
#SBATCH -J Inference
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=2:00:00
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
fid_eval=0

dataset_name="ImageNet"
task="sr"

sigma=0.01
num_timesteps=3
max_images=3
max_unfolded_iter=3
kernel_size=5
lamb=0.3

for num_timesteps in 3 ; do
    learned_lambda_sig=False
    general=False

    if [ "$general" = "True" ]; then
        ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_generalised_3_rank_5/imagenet/sigma_model_0.01/03-06-2025"
        ckpt_name="imagenet_BestPSNR_lora_epoch=087-5.ckpt"
        op="gaussian"
    else
        if [ "$task" = "inp" ]; then
            ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_3_rank_5/imagenet/sigma_model_0.01/inp/random/19-06-2025/scale_0.3"
            ckpt_name="imagenet_BestPSNR_lora_inp_random_epoch=027-5.ckpt"
            op="random"
            # ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_3_rank_5/lsun/sigma_model_0.025/inp/box/25-06-2025/scale_0.2"
            # ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpt_adv_test30/lsun/inp/box/18-06-2025/scale_0.5/noise_var_0.05"
            # ckpt_name="lsun_BestPSNR_lora_inp_box_epoch=055_5.ckpt"
            # ckpt_name="lsun_lora_inp_box_epoch=079-5.ckpt"
            # op="box"
        elif [ "$task" = "sr" ]; then
            # ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_3_rank_5/lsun"
            # ckpt_name="LSUN_RAM_SRX4.ckpt"
            # ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_1_rank_5/lsun/sigma_model_0.01/sr/gaussian_3/19-06-2025/factor_4"
            # ckpt_name="lsun_BestPSNR_lora_sr_gaussian_epoch=087-5.ckpt"
            ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_3_rank_5/lsun/sigma_model_0.01/sr/gaussian_3/21-06-2025/factor_4"
            ckpt_name="lsun_BestPSNR_lora_sr_gaussian_epoch=111-5.ckpt"

            ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_3/imagenet/sigma_model_0.01/sr/gaussian/24-05-2025/factor_4"
            ckpt_name="imagenet_BestPSNR_lora_sr_epoch=254.ckpt"
            op="gaussian"
        elif [ "$task" = "deblur" ]; then
            ckpt_dir="/users/cmk2000/sharedscratch/CKM/Regularised_CDM/ckpts_RAM_unfolded_3/imagenet/sigma_model_0.01/deblur/gaussian/23-05-2025/kernel_size_25"
            ckpt_name="imagenet_BestPSNR_lora_deblur_epoch=140.ckpt"
            op="gaussian"
        else
            echo "Task not supported yet!"
        fi
    fi

    if [ "$dataset_name" = "ImageNet" ]; then
        config_file="imagenet_configs.yaml"
    else
        config_file="fine_tuning_lmodel.yaml"
    fi

    if [ $fid_eval -eq 0 ]; then
        python3 sampling.py \
                        --task $task \
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
                        --learned_lambda_sig $learned_lambda_sig \
                        --use_RAM True \
            > "$RESULTS_DIR"_out_sample_dif.output \
            2> "$RESULTS_DIR"_err_sample_dif.error
        
    elif [ $fid_eval -eq 1 ]; then
        echo "Computing FID: lambda: $lamb and sigma $sigma on dataset_name $dataset_name"          
        python3 fid.py \
                -d "/users/cmk2000/sharedscratch/CKM/Regularised_CDM/Results_RAM_unfolded_iter_1_rank_5_paper/LSUN/sr_lora/operator_gaussian_5/var_0.01/20-06-2025/ddim/factor_4/timesteps_27/zeta_0.9/eta_0.8/lambda_0.31" \
            > "$RESULTS_DIR"_out_fid.output \
            2> "$RESULTS_DIR"_err_fid.error
    else
        echo "Next..."
    fi
done
# Final message
echo "Finish!!"
