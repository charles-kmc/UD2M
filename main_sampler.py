import os
import copy
import datetime
import torch
from torch.utils.data import DataLoader

from unfolded_models.unfolded_model import (
    HQS_models, 
    physics_models, 
    DiffusionScheduler,
    GetDenoisingTimestep,
)
from unfolded_models.ddim_sampler import (
    load_trainable_params,
    DDIM_SAMPLER,
    load_yaml,
)
from args_unfolded_model import args_unfolded
from models.load_frozen_model import load_frozen_model
from datasets.datasets import Datasets
from utils_lora.lora_parametrisation import LoRa_model
import deepinv

def main():
    # args
    args = args_unfolded()
    
    # parameters
    LORA = True
    device = deepinv.utils.get_freer_gpu() #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("00000000", device,  torch.cuda.is_available())
    print(torch.cuda.is_available())  # Should return True if CUDA is available
    
    print(torch.cuda.current_device())  # Should return the index of the current GPU
    print(torch.cuda.device_count())  # Should return the number of GPUs available
    print(torch.cuda.get_device_name(0))
    
    # data
    dir_test = "/users/cmk2000/sharedscratch/Datasets/testsets/ffhq"
    datasets = Datasets(
        dir_test, 
        args.im_size, 
        dataset_name=args.dataset_name
    )
    
    testset = DataLoader(
        datasets, 
        batch_size=1, 
        shuffle=False, 
    )
    
    # model
    frozen_model_name = 'diffusion_ffhq_10m' 
    frozen_model_dir = '/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo' 
    frozen_model_path = os.path.join(frozen_model_dir, f'{frozen_model_name}.pt')
    frozen_model, _ = load_frozen_model(frozen_model_name, frozen_model_path, device)
    
    # LoRA model 
    model = copy.deepcopy(frozen_model)
    
    args.mode = "eval"
    args.date = "08-10-2024"
    args.epoch = 120
    
    if LORA:
        LoRa_model(
            model, 
            target_layer = ["qkv", "proj_out"], 
            device=device, 
            rank = args.rank
        )
        load_trainable_params(model, args.epoch, args, device)
        
    # physics
    physics = physics_models(
        args.physic.kernel_size, 
        args.physic.blur_name, 
        device = device,
        sigma_model = args.physic.sigma_model,
        transform_y= args.physic.transform_y
    )
    # HQS model
    denoising_timestep = GetDenoisingTimestep(device)
    diffusion = DiffusionScheduler(device=device)
    hqs_model = HQS_models(
        model,
        physics, 
        diffusion, 
        denoising_timestep,
        args
    )
    # sampler
    diffsampler = DDIM_SAMPLER(
        hqs_model, 
        physics, 
        diffusion, 
        device, 
        args
    )

    # parameters
    num_timesteps = 100
    eta = 0.0
    zeta = 0.4
    args.zeta = zeta
    args.eta = eta
    for ii, im in enumerate(testset):
        # obs
        im = im.to(device)
        y = physics.y(im)
        #sampling
        im_name = f"_{(ii):05d}"
        diffsampler.sampler(
            im,
            y, 
            im_name, 
            num_timesteps = num_timesteps, 
            eta=eta, 
            zeta=zeta
        )
    
if __name__ == "__main__":
    main()