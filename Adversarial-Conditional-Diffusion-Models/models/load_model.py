#!/usr/bin/env python3
import os
import copy
import torch
import blobfile as bf
from guided_diffusion import dist_util
from utils import utils_model
import utils as utils
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def load_frozen_model(model_name, model_checkpoint_path):
    # load model
    model_config = dict(
            model_checkpoint_path=model_checkpoint_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_checkpoint_path=model_checkpoint_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    
    # agruments for models   
    args = utils_model.create_argparser(model_config).parse_args([])
    
    # create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_checkpoint_path, map_location="cpu")
    )
    
    # set model to eval mode
    model.train()
    for _k, v in model.named_parameters():
        v.requires_grad = False
    return model, diffusion

# load lora weights                
def load_trainable_params(model, args):
    """_summary_
    """
    device = args.device
    lora_checkpoint_dir = args.lora_checkpoint_dir
    lora_checkpoint_name = args.lora_checkpoint_name
    filepath = bf.join(lora_checkpoint_dir, lora_checkpoint_name)
    trainable_state_dict = torch.load(filepath, map_location=device)
    try:
        model.load_state_dict(trainable_state_dict["model_state_dict"], strict=False)  
    except:
        model.load_state_dict(trainable_state_dict["state_dict"], strict=False)  

    # set model to eval mode
    model.eval()
    for _k, v in model.named_parameters():
        v.requires_grad = False    
    model = model.to(device)         
    return model

def adapter_lora_model(args):
    """_summary_
    """
    frozen_model_path = os.path.join(args.model.frozen_model_dir, f'{args.model.frozen_model_name}.pt')
    model_frozen, _ = load_frozen_model(args.model.frozen_model_name, frozen_model_path)
    
    # ---> LoRA configuration
    model = copy.deepcopy(model_frozen)
    
    if args.model.use_lora:
        target_modules = ["qkv", "proj_out"]
        utils.LoRa_model(
                        model, 
                        target_layer = target_modules, 
                        rank = args.model.rank
                    )
        
        # ---> Gradient activation 
        for name, params in model.named_parameters():
            if "lora" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
                
        for param in model.out.parameters():
            param.requires_grad_(True)
    
        # for name, param in model.named_parameters():
        #     if "output_blocks.2.0.out_layers.3" in name:
        #         param.requires_grad_(True)
    
    return model