import torch 
import torch.nn as nn
import torch.nn.functional as F

import os
import copy

from models.model import FineTuningDiffusionModel
from models.load_pre_trained_models import load_pre_trained_model
from utils_lora.lora_parametrisation import LoRa_model
from models import dist_util, logger 
from datasets.datasets import get_data_loader
from models.trainer import Trainer

def main():
    
    # --- device
    device = dist_util.dev()
    
    # --- Pre trained model 
    pre_trained_model_name = 'diffusion_ffhq_10m' 
    pre_trained_dir = '/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/pretrained_models/model_zoo'  
    pre_trained_model_path = os.path.join(pre_trained_dir, f'{pre_trained_model_name}.pt')
    pre_trained_model, diffusion = load_pre_trained_model(
                                            pre_trained_model_name, 
                                            pre_trained_model_path, 
                                            device
                                        )
    pre_trained_model_copy = copy.deepcopy(pre_trained_model)
    
    # LoRa model from the pretrained model
    input_block_layer_lora = [9]
    middle_block_layer_lora = [1]
    output_block_layer_lora = [2, 3]
    lora_pre_trained_model = LoRa_model(
                                pre_trained_model_copy, 
                                input_block_layer_lora = input_block_layer_lora, 
                                middle_block_layer_lora = middle_block_layer_lora, 
                                output_block_layer_lora = output_block_layer_lora,
                                rank = 2,
                                device = device
                            )

    # Augmented LoRa model
    image_size = 256
    in_channels = 3
    model_channels = 128
    out_channels = 3
    aug_LoRa_model = FineTuningDiffusionModel(
                                image_size, 
                                in_channels, 
                                model_channels, 
                                out_channels, 
                                lora_pre_trained_model
                            )
    
    print(aug_LoRa_model.device)

    # --- detaset 
    dataset_dir = ""
    batch_size = 128
    train_dataloader = get_data_loader(dataset_dir, image_size, batch_size)
    
    # --- trainer
    num_epochs = 1000
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
                resume_checkpoint = "",
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