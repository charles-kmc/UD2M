
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
import yaml
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import functools
import diffusers    
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import time

gradient_checkpointing = True

accelerator = Accelerator(
    gradient_accumulation_steps=...,
    mixed_precision=...,
    log_with=...,
    project_config=...,
    kwargs_handlers=...,
)

# TODO --> set seed

# TODO -->load unet
unet = UNet2DConditionModel(
    subfolder=...,
    revision=...,
    variant=...,
)

unet.requires_grad_(False)

weight_dtype = torch.float32

unet.to(accelerator.device, dtype = weight_dtype)

unet_lora_config = LoraConfig(
    r=...,
    lora_alpha=...,
    init_lora_weights=...,
    target_modules=...,
)

# Applying Lora to the unet
unet.add_adapter(unet_lora_config)
unet.to(accelerator.device, dtype = weight_dtype)

# make sure trainable parameters are float 32
cast_training_params(unet, dtype = weight_dtype)

# TODO --> Gradient checkpointing
if gradient_checkpointing:
    unet.enable_gradient_checkpointing()

learning_rate = ...

params_to_optimize = list(filter(lambda p: p.required_grad, unet.parameters()))
optimizer_class = torch.optim.adamw
optimizer = optimizer_class(params_to_optimize,
                            lr = learning_rate,
                            betas = ...,
                            weight_decay=...,
                            eps = ...
                            )


