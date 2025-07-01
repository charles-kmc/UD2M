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
    create_classifier,
)

import lsun_diffusion.pytorch_diffusion as lsun_diffusion

def load_frozen_model(model_name, model_checkpoint_path):
    # load model
    if model_name == "CIFAR10_diff":
        model_config = dict(
            model_checkpoint_path=model_checkpoint_path,
            num_channels=128,
            num_res_blocks=2,
            channel_mult = "1,2,2,2",
            attention_resolutions="16",
            image_size = 32,
        )

    elif model_name == "diffusion64":
        model_config = dict(
            model_checkpoint_path=model_checkpoint_path,
            num_channels=192,
            num_res_blocks=3,
            attention_resolutions="32,16,8",
            channel_mult = "1,2,3,4",
            class_cond = True,
            use_new_attention_order = True,
            resblock_updown = True,
            image_size = 64,
            noise_schedule = "cosine",
            use_fp16 = False,
        )
    
    elif model_name == "lsun_bedroom":
        model_config = dict(
            model_checkpoint_path=model_checkpoint_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="32,16,8",
            image_size = 256,
            class_cond = False,
            learn_sigma = True,
            noise_schedule = "linear",
            num_head_channels = 64,
            use_fp16 = False,
            use_scale_shift_norm = True,
            dropout = 0.1,
            use_new_attention_order = True,
            resblock_updown = True,
        )

    else:
        model_config = dict(
                model_checkpoint_path=model_checkpoint_path,
                num_channels=128,
                num_res_blocks=1,
                attention_resolutions="16",
            ) if model_name == 'diffusion_ffhq_10m'\
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
    # model.num_classes = None
    
    # set model to eval mode
    model.train()
    for _k, v in model.named_parameters():
        v.requires_grad = False
    return model, diffusion

def load_classifier(model_name, model_checkpoint_path):
    if model_name == "diffusion64":
        model_config = dict(
            image_size = 64,
            classifier_use_fp16  =False,
            classifier_width = 128,
            classifier_depth = 4,
            classifier_attention_resolutions = "132,16,8",
            classifier_channel_mult = "1,2,3,4",
            classifier_use_scale_shift_norm = True,
            classifier_resblock_updown = True,
            classifier_pool  = "attention",
        )
    
    classifier = create_classifier(**model_config)
    classifier.load_state_dict(
        dist_util.load_state_dict(model_checkpoint_path, map_location="cpu")
    )
    for k, v in classifier.named_parameters():
        v.requires_grad = False
    return classifier
    
    


# -- LSUN pre-trained model
def load_frozen_model_lsun(model_checkpoint_path):
    # load model
    lsun_cfg = {
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1,1,2,2,4,4),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.0,
        }
    model = lsun_diffusion.Model(**lsun_cfg)
    #print(model)
    # model = lsun_diffusion.Model(resolution=32,
    #             in_channels=3,
    #             out_ch=3,
    #             ch=128,
    #             ch_mult=(1,2,2,2),
    #             num_res_blocks=2,
    #             attn_resolutions=(16,),
    #             dropout=0.1)
    
    model.load_state_dict(
        dist_util.load_state_dict(model_checkpoint_path, map_location="cpu")
    )
    # set model to eval mode
    model.train()
    for _k, v in model.named_parameters():
        v.requires_grad = False
    return model


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
    # pre-trained model 
    if args.data.dataset_name=="LSUN":
        model_name = "lsun_bedroom" #"ema_lsun_bedroom"
        frozen_model_path = lsun_diffusion.get_ckpt_path(model_name, root=args.model.frozen_model_dir)
        model_frozen = load_frozen_model_lsun(frozen_model_path)
        target_modules = [".q", ".k", ".v", "proj_out"] 
        # frozen_model_name = "lsun_bedroom"
        # frozen_model_path = os.path.join(args.model.frozen_model_dir, f'{frozen_model_name}.pt')
        # model_frozen, _ = load_frozen_model(frozen_model_name, frozen_model_path)
        # target_modules = ["qkv", "proj_out"]
    elif args.data.dataset_name in ["FFHQ", "ImageNet"]:
        try:
            frozen_model_name = 'diffusion_ffhq_10m' #if args.data.dataset_name=="FFHQ" else '256x256_diffusion_uncond'
            frozen_model_path = os.path.join(args.model.frozen_model_dir, f'{frozen_model_name}.pt')
            model_frozen, _ = load_frozen_model(frozen_model_name, frozen_model_path)
        except:
            frozen_model_name = 'diffusion_ffhq_10m' 
            frozen_model_path = os.path.join(args.model.frozen_model_dir, f'{frozen_model_name}.pt')
            model_frozen, _ = load_frozen_model(frozen_model_name, frozen_model_path)
        target_modules = ["qkv", "proj_out"]
    elif args.data.dataset_name=="Imagenet64":
        frozen_model_name = 'diffusion64' 
        frozen_model_path = os.path.join(args.model.frozen_model_dir, f'{frozen_model_name}.pt')
        model_frozen, _ = load_frozen_model(frozen_model_name, frozen_model_path)
        target_modules = ["qkv", "proj_out"]
    else:
        raise ValueError(f"We don't have a pre-trained model for this dataset: {args.dataset_name}")
   
    # ---> LoRA adapter
    model = copy.deepcopy(model_frozen)
    if args.model.use_lora:
        utils.LoRa_model(model, target_layer=target_modules, rank=args.model.rank)
        # ---> Gradient activation 
        for name, params in model.named_parameters():
            if "lora" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        # Last layer
        if args.data.dataset_name=="LSUN":
            for param in model.conv_out.parameters():
                param.requires_grad_(True)
        elif args.data.dataset_name in ["FFHQ", "ImageNet64", "ImageNet"]:
            for param in model.out.parameters():
                param.requires_grad_(True)
    return model