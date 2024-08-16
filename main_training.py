#!/usr/bin/env python3

import torch #type:ignore
import torch.nn as nn#type:ignore
import torch.nn.functional as F #type:#type:ignore
from torchvision.utils import save_image#type:ignore

import os
import copy

from models.model import FineTuningDiffusionModel
from models.load_frozen_model import load_frozen_model
from utils_lora.lora_parametrisation import LoRa_model
from models import dist_util, logger 
from datasets.datasets import get_data_loader
from models.trainer import Trainer
import datetime 

def main():
    date = datetime.datetime.now().strftime("%d-%m-%Y")

    # path for resuming checkpoints during training
    save_checkpoint_dir = os.path.join("/users/cmk2000/sharedscratch/Pretrained-Checkpoints/conditional-diffusion_model-for_ivp", date)
    save_results = os.path.join("/users/cmk2000/sharedscratch/Results/conditional-diffusion_model-for_ivp", date)
    
    # --- device
    dist_util.setup_dist()
    device = dist_util.dev() #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Pre trained model 
    frozen_model_name = 'diffusion_ffhq_10m' 
    frozen_model_dir = '/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo' 
    frozen_model_path = os.path.join(frozen_model_dir, f'{frozen_model_name}.pt')
    frozen_model, diffusion = load_frozen_model(frozen_model_name, frozen_model_path, device)

    
    # LoRa model from the pretrained model
    lora_pre_trained_model = LoRa_model(frozen_model, device, rank = 2)

    # Augmented LoRa model
    image_size = 256
    in_channels = 3
    model_channels = 128
    out_channels = 3
    aug_LoRa_model = FineTuningDiffusionModel(image_size, in_channels, model_channels, out_channels, device, frozen_model = lora_pre_trained_model)
    
    print("device", aug_LoRa_model.device, device)

    # --- detaset 
    dataset_dir = "/users/cmk2000/sharedscratch/Datasets/ImageNet/train"
    batch_size = 8
    prop = 0.3
    train_dataloader = get_data_loader(dataset_dir, image_size, batch_size, prop)
    # data_iter = iter(train_dataloader)
    # ref, deg, op = next(data_iter)
    
    print(len(train_dataloader))
       
    # --- trainer
    num_epochs = 1000
    lr = 1e-3
    microbatch = 0
    ema_rate = 0.9999
    log_interval = 100
    save_interval = 100
    resume_checkpoint = ""
    problem_type = "deblur"
    trainer = Trainer(
                aug_LoRa_model, 
                diffusion,
                train_dataloader,
                lr=lr, 
                batch_size=batch_size,
                save_checkpoint_dir = save_checkpoint_dir,
                save_results = save_results,
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