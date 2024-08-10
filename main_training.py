#!/usr/bin/env python3

import torch 
import torch.nn as nn
import torch.nn.functional as F #type:

import os
import copy

from models.model import FineTuningDiffusionModel
from models.load_frozen_model import load_frozen_model
from utils_lora.lora_parametrisation import LoRa_model
from models import dist_util, logger 
from datasets.datasets import get_data_loader
from models.trainer import Trainer

def main():
    # path for resuming checkpoints during training
    resume_checkpoint_dir = "/users/cmk2000/sharedscratch/Pretrained-Checkpoints/conditional-diffusion_model-for_ivp"
    
    # --- device
    device = dist_util.dev()
    
    # --- Pre trained model 
    frozen_model_name = 'diffusion_ffhq_10m' 
    frozen_model_dir = '/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo'  
    frozen_model_path = os.path.join(frozen_model_dir, f'{frozen_model_name}.pt')
    frozen_model, diffusion = load_frozen_model(frozen_model_name, frozen_model_path, device)

    
    # LoRa model from the pretrained model
    lora_pre_trained_model = LoRa_model(frozen_model, rank = 2, device = device)

    # Augmented LoRa model
    image_size = 256
    in_channels = 3
    model_channels = 128
    out_channels = 3
    aug_LoRa_model = FineTuningDiffusionModel(image_size, in_channels, model_channels, out_channels, lora_pre_trained_model)
    
    print(aug_LoRa_model.device)

    # --- detaset 
    dataset_dir = "/users/cmk2000/sharedscratch/Datasets/ImageNet/train"
    batch_size = 128
    train_dataloader = get_data_loader(dataset_dir, image_size, batch_size)
    
    # --- trainer
    num_epochs = 100
    lr = 1e-3
    save_model = 100
    batch_size = 32
    microbatch = 0
    ema_rate = 0.9999
    log_interval = 100
    save_interval = 100
    problem_type = "deblur"
    trainer = Trainer(
                aug_LoRa_model, 
                diffusion,
                train_dataloader,
                lr=lr, 
                batch_size=batch_size,
                save_model=save_model,
                resume_checkpoint = resume_checkpoint_dir,
                log_interval=log_interval,
                save_interval=save_interval,
                ema_rate=ema_rate,
                microbatch=microbatch,
                problem_type = problem_type,
            )

    # --- run training loop
    trainer.run_training_loop(num_epochs)
    
if __name__ == "__main__":
    main()