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
#SBATCH -J Inference
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=00:30:00
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
RESULTS_DIR1=""
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
        ckpt_dir=directory-to-checkpoint
        ckpt_name=checkpoint-name
        op="gaussian"
    else
        if [ "$task" = "inp" ]; then
            ckpt_dir=directory-to-checkpoint
            ckpt_name=checkpoint-name
            op="random" # random, box
            
        elif [ "$task" = "sr" ]; then
            ckpt_dir=directory-to-checkpoint
            ckpt_name=checkpoint-name
            op="gaussian"            # gaussian, uniform, motion
        elif [ "$task" = "deblur" ]; then
            ckpt_dir=directory-to-checkpoint
            ckpt_name=checkpoint-name
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
                -d path-to-results-and-ground-truth \
            > "$RESULTS_DIR"_out_fid.output \
            2> "$RESULTS_DIR"_err_fid.error
    else
        echo "Next..."
    fi
done
# Final message
echo "Finish!!"
