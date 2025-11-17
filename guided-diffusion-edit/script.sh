#!/bin/bash
# Add date and time to save directory
DATE=$(date +%Y-%m-%d_%H-%M-%S)
# Model="ema_0.9999_"
Model="model"
CKPT="030000"
export OPENAI_LOGDIR=./logs/Samples_${Model}_${CKPT}
MODEL_FLAGS="--image_size 32 --num_channels 64 --num_res_blocks 2 --grayscale True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--batch_size 64 --lr 1e-4 --log_interval 100"
SAMPLE_FLAGS="--batch_size 4 --num_samples 16 --timestep_respacing 250"
# python scripts/image_train.py --data_dir ./data/mnist_images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

python scripts/image_sample.py --model_path logs/2025-10-02_16-26-53/${Model}${CKPT}.pt $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
python scripts/proc_ims.py --im_file $OPENAI_LOGDIR/samples_16x32x32x1.npz