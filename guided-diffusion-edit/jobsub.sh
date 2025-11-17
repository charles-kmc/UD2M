#!/bin/bash -l

################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D "/users/js3006/conddiff/UD2M/guided-diffusion-edit"
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
##SBATCH -o job-%j.output
#SBATCH -o o_train_dif.output
#SBATCH -e e_train_dif.error
## Job name
#SBATCH -J CDMablation
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=1:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem=16000
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
flight env activate gridware
module load apps/anaconda3
module load apps/nvidia-cuda/11.2.2
source activate base
conda activate Cond_diff

#===========================
#  Create results directory
#---------------------------
RESULTS_DIR1="/users/js3006/Conditional-Diffusion-Models-for-IVP/Adversarial-Conditional-Diffusion-Models/CDM-outputs/${SLURM_JOB_NAME}"
RESULTS_DIR=$RESULTS_DIR1"/${SLURM_JOB_ID}"
echo "Your results will be stored in: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR1"

#===============================
#  Application launch commands
#-------------------------------
# Running python scripts
# DATE=$(date +%Y-%m-%d_%H-%M-%S)
# export OPENAI_LOGDIR=./logs/$DATE
# MODEL_FLAGS="--image_size 32 --num_channels 64 --num_res_blocks 2 --grayscale True"
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
# TRAIN_FLAGS="--batch_size 64 --lr 1e-4 --log_interval 100 --save_interval 5000"

# python scripts/image_train.py --data_dir ./data/mnist_images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

Model="model"
CKPT="220000"
export OPENAI_LOGDIR=./logs/Samples_${Model}_${CKPT}
MODEL_FLAGS="--image_size 32 --num_channels 64 --num_res_blocks 2 --grayscale True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--batch_size 64 --lr 1e-4 --log_interval 100"
SAMPLE_FLAGS="--batch_size 4 --num_samples 16 --timestep_respacing 250"
# python scripts/image_train.py --data_dir ./data/mnist_images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

python scripts/image_sample.py --model_path logs/2025-10-02_16-26-53/${Model}${CKPT}.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
python scripts/proc_ims.py --im_file $OPENAI_LOGDIR/samples_16x32x32x1.npz

Model="ema_0.9999_"
CKPT="220000"
export OPENAI_LOGDIR=./logs/Samples_${Model}_${CKPT}
MODEL_FLAGS="--image_size 32 --num_channels 64 --num_res_blocks 2 --grayscale True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--batch_size 64 --lr 1e-4 --log_interval 100"
SAMPLE_FLAGS="--batch_size 4 --num_samples 16 --timestep_respacing 250"
# python scripts/image_train.py --data_dir ./data/mnist_images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

python scripts/image_sample.py --model_path logs/2025-10-02_16-26-53/${Model}${CKPT}.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
python scripts/proc_ims.py --im_file $OPENAI_LOGDIR/samples_16x32x32x1.npz