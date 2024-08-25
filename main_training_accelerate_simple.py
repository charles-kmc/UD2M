#!/usr/bin/env python3

import os
import copy

from models.load_frozen_model import load_frozen_model
from models.pipeline import Pipeline
from utils_lora.lora_parametrisation import LoRa_model
from models import dist_util 
from datasets.datasets import get_data_loader
from models.trainer_accelerate_simple import Trainer_accelerate_simple
from utils.get_logger import get_loggers
import datetime 

#torch.backends.cudnn.benchmark = True

def main():
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    script_dir = os.path.dirname(__file__)
    script_path = script_dir.rsplit("/", 1)[0]
    log_dir = os.path.join(script_path, "logs", "Conditional-Diffusion-Models-for-IVP", "Loggers", date)
    os.makedirs(log_dir, exist_ok=True)
    loggers = get_loggers(log_dir, f"training_log_simple.log")
    
    prop = 0.1
    
    # path for resuming checkpoints during training
    save_checkpoint_dir = os.path.join("/users/cmk2000/sharedscratch/Pretrained-Checkpoints/conditional-diffusion_model-for_ivp", date, f"prop_data_{prop}", "simple")
    save_results = os.path.join("/users/cmk2000/sharedscratch/Results/conditional-diffusion_model-for_ivp", date, f"prop_data_{prop}", "simple")
    
    # --- device
    dist_util.setup_dist()
    device = dist_util.dev() 
    
    # --- Pre trained model 
    frozen_model_name = 'diffusion_ffhq_10m' 
    frozen_model_dir = '/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo' 
    frozen_model_path = os.path.join(frozen_model_dir, f'{frozen_model_name}.pt')
    frozen_model, diffusion = load_frozen_model(frozen_model_name, frozen_model_path, device)

    # LoRa model from the pretrained model
    lora_model = copy.deepcopy(frozen_model)
    LoRa_model(frozen_model, device, rank = 5)
    lora_model = Pipeline(lora_model, device)
    
    # Frozen ema model
    ema_lora_model = copy.deepcopy(lora_model)
    for param in ema_lora_model.parameters():
        param.requires_grad = False
        
    # Layer =s to fine tuning
    for param in lora_model.l_model.out.parameters():
        param.requires_grad_(True)
    for name, param in lora_model.named_parameters():
        if "output_blocks.2.0.out_layers.3" in name:
            param.requires_grad_(True)
    
    # Augmented LoRa model 
    image_size = 256
    loggers.info(f"device model: {lora_model.device} <--> {device}")

    # --- detaset 
    dataset_dir = "/users/cmk2000/sharedscratch/Datasets/ImageNet/train"
    batch_size = 8
    train_dataloader, testloader = get_data_loader(dataset_dir, image_size, batch_size, prop)
    loggers.info(f"number of batch in dataset: {len(train_dataloader)}")
       
    # --- trainer
    num_epochs = 100
    lr = 1e-3
    ema_rate = 0.99
    log_interval = 20
    save_interval = 5
    problem_type = "deblur"
    resume_checkpoint = ""#"/users/cmk2000/sharedscratch/Pretrained-Checkpoints/conditional-diffusion_model-for_ivp/24-08-2024/prop_data_0.1/simple"
    
    trainer_accelerate_simple = Trainer_accelerate_simple(
                lora_model, 
                ema_lora_model,
                diffusion,
                train_dataloader,
                testloader,
                loggers,
                lr=lr, 
                batch_size=batch_size,
                save_checkpoint_dir = save_checkpoint_dir,
                save_results = save_results,
                log_interval=log_interval,
                save_interval=save_interval,
                ema_rate=ema_rate,
                problem_type = problem_type,
                resume_checkpoint = resume_checkpoint
            )

    # --- run training loop
    loggers.info("Start training ...")
    trainer_accelerate_simple.run_training_loop(num_epochs)
    loggers.info("Training finish !!")

if __name__ == "__main__":
    main()