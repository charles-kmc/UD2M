import os
import copy
import datetime
import torch
from torch.utils.data import DataLoader, random_split

from args_unfolded_model import args_unfolded
from models.load_model import load_frozen_model
from datasets.datasets import GetDatasets
from configs.args_parse import configs

import unfolded_models as um
import physics as phy
import utils as utils

def main():
    # args
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    args = args_unfolded()
    config = configs()
    args.task = config.task
    args.physic.operator_name = config.operator_name
    if config.task == "inp":
        args.dpir.use_dpir = config.use_dpir
    args.lambda_ = config.lambda_
    args.date = date
    args.max_unfolded_iter = 3
    
    # logger 
    script_dir = os.path.dirname(__file__)
    script_path = script_dir.rsplit("/", 1)[0]
    log_dir = os.path.join(script_path, "logs", "Logger_CDM", date)
    os.makedirs(log_dir, exist_ok=True)
    logger = utils.get_loggers(log_dir, f"log_unfolded.log")
    
    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # datasets
    prop = 0.2
    datasets = GetDatasets(
        args.dataset_path, 
        args.im_size, 
        dataset_name=args.dataset_name
    )
    train_size = int(prop * len(datasets))
    test_size = len(datasets) - train_size
    train_dataset, test_dataset = random_split(
        datasets, 
        [train_size, test_size]
    )
    trainloader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers
    )
    testloader = DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers
    )
    
    # model 
    frozen_model_name = 'diffusion_ffhq_10m' 
    frozen_model_dir = '/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo' 
    frozen_model_path = os.path.join(frozen_model_dir, f'{frozen_model_name}.pt')
    frozen_model, _ = load_frozen_model(frozen_model_name, frozen_model_path, device)
    
    # LoRA model 
    model = copy.deepcopy(frozen_model)
    target_modules = ["qkv", "proj_out"]
    utils.LoRa_model(
        model, 
        target_layer =target_modules, 
        device=device, 
        rank = args.rank
    )
    for param in model.out.parameters():
        param.requires_grad_(True)
    for name, param in model.named_parameters():
        if "output_blocks.2.0.out_layers.3" in name:
            param.requires_grad_(True)
    
    # Diffusion noise
    diffusion_scheduler = um.DiffusionScheduler(device=device,noise_schedule_type=args.noise_schedule_type)
    
    # physic
    if args.task == "deblur":
        kernels = phy.Kernels(
            operator_name=args.physic.operator_name,
            kernel_size=args.physic.kernel_size,
            device=device,
            mode = args.mode
        )
        physic = phy.Deblurring(
            sigma_model=args.physic.sigma_model,
            device=device,
            transform_y=args.physic.transform_y,
        )
        physics = utils.DotDict({"kernels": kernels, "physic": physic})
    elif args.task == "inp":
        physic = phy.Inpainting(
            mask_rate=args.physic.inp.mask_rate,
            im_size=args.im_size,
            operator_name=args.physic.operator_name,
            box_proportion = args.physic.inp.box_proportion,
            index_ii = args.physic.inp.index_ii,
            index_jj = args.physic.inp.index_jj,
            device=device,
            sigma_model=args.physic.sigma_model,
            transform_y=args.physic.transform_y,
            mode=args.mode,
        )
        physics = utils.DotDict({"kernels": None, "physic": physic})
        
    # HQS module
    denoising_timestep = um.GetDenoisingTimestep(device)
    hqs_module = um.HQS_models(
        model,
        physic, 
        diffusion_scheduler, 
        denoising_timestep,
        args
    )
    
    # trainer module
    
    trainer = um.Trainer(
        model,
        diffusion_scheduler=diffusion_scheduler,
        physic=physics.physic,
        kernels=kernels,
        hqs_module = hqs_module,
        trainloader=trainloader,
        testloader=testloader,
        logger=logger,
        device=device,
        args=args,
        max_unfolded_iter=args.max_unfolded_iter
    )
    
    # training loop
    trainer.training(epochs=10000)
    
if __name__ == "__main__":
    main()