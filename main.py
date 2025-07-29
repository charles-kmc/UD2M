import os
import copy
import datetime
import yaml
import wandb
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split, RandomSampler
import torch.profiler as tp
from models.load_model import load_frozen_model
from datasets.datasets import GetDatasets
from datasets.get_datasets import get_dataset, get_dataloader
from configs.args_parse import configs
import unfolded_models as um
import physics as phy
import utils as utils
from lsun_diffusion.pytorch_diffusion.ckpt_util import get_ckpt_path

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

def main():
    # root and date
    script_dir = os.path.dirname(__file__)
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    
    # args
    config = configs()
    with open(os.path.join(script_dir, "configs", config.config_file), 'r') as file:
        config_dict = yaml.safe_load(file)
    
    args = utils.dict_to_dotdict(config_dict)
    args.data.dataset_name = config.dataset_name
    args.physic.operator_name = config.operator_name
    args.learned_lambda_sig = config.learned_lambda_sig
    args.physic.sigma_model = config.sigma_model
    args.max_unfolded_iter = config.max_unfolded_iter
    args.physic.kernel_size = config.kernel_size
    args.lambda_ = config.lambda_
    args.use_RAM = config.use_RAM
    args.task = config.task
    args.date = date

    # logger 
    log_dir = os.path.join(script_dir, f"Logs", date)
    os.makedirs(log_dir, exist_ok=True)
    logger = utils.get_loggers(log_dir, f"ud2m.log")
    
    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # train 
    dataset_train = get_dataset(
        name=args.data.dataset_name, 
        root_dataset=args.data.root, 
        im_size=args.data.im_size, 
        subset_data="train",
    )
    loader_train = get_dataloader(dataset_train,args=args)
    
    # validation
    dataset_val = get_dataset(
        name=args.data.dataset_name, 
        root_dataset=args.data.root, 
        im_size=args.data.im_size, 
        subset_data="val",
    )
    loader_val = get_dataloader(dataset_val,args=args, subset_data="val")
    
    # physic
    os.makedirs(os.path.join(args.save_checkpoint_dir, 'wandb', date), exist_ok=True)
    
    if args.task == "" or args.physic.operator_name == "":
        wandb_logger = WandbLogger(
            project=f"Adversarial Conditional diffusion models general operators {args.data.dataset_name}",  
            name=date+"_lambda"+str(args.lambda_),
            log_model=False,
            dir=os.path.join(args.save_checkpoint_dir, 'wandb', date),
        )
    else:
        if args.task in ["deblur", "sr"]:
            physic_configs = {
                "operator_name": args.physic.operator_name, 
            }
            if args.task == "sr":
                physic_configs["sf"] = args.physic.sr.sf
        elif args.task == "inp":
            physic_configs = {
                "mask_rate": args.physic.inp.mask_rate,
                "operator_name": args.physic.operator_name,
                "box_proportion": args.physic.inp.box_proportion,
                "index_ii": args.physic.inp.index_ii,
                "index_jj": args.physic.inp.index_jj,
            }
        elif args.task == "jpeg":
            physic_configs = {
                "qf": args.physic.jpeg.qf,
            }
        else:
            physic_configs = {}
        physic_configs.update({
            "im_size": args.data.im_size,
            "sigma_model": args.physic.sigma_model,
            "scale_image": args.physic.transform_y,
            "mode": args.mode,
            "device": device,
        })
        kernels_configs = {
            "operator_name": args.physic.operator_name,
            "kernel_size": args.physic.kernel_size, 
            "device": device, 
            "mode": args.mode
        }
        kernels = phy.Kernels(**kernels_configs)
        if args.task in ["deblur", "sr", "inp", "phase_retrieval", "jpeg"]:
            physic = phy.get_operator(name=args.task, **physic_configs)
        else:
            raise ValueError(f"This task ({args.task}) is not implemented!!!")
    
        if args.task == "deblur":
            from deepinv.physics import BlurFFT, GaussianNoise
            dinv_physic = BlurFFT(
                (3,256,256),
                filter = kernels.get_blur().unsqueeze(0).unsqueeze(0),
                device = device,
                noise_model = GaussianNoise(
                    sigma= args.physic.sigma_model,
                ),
            )
        elif args.task == "inp":
            from deepinv.physics import Inpainting, GaussianNoise
            dinv_physic = Inpainting(
                (3,256,256),
                mask = physic.Mask,
                device = device,
                noise_model = GaussianNoise(
                    sigma= args.physic.sigma_model,
                ),
            ).to(device)
        elif args.task == "sr":
            from deepinv.physics import Downsampling, GaussianNoise
            dinv_physic =   Downsampling(
                filter = "bicubic",
                img_size = (3, 256, 256),
                factor = args.physic.sr.sf,
                device = device,
                noise_model = GaussianNoise(
                    sigma= args.physic.sigma_model,
                ),
            ).to(device)
        else:
            raise ValueError(f"This task ({args.task})is not implemented!!!")
    
        wandb_logger = WandbLogger(
            project=f"UD2M_ram{args.use_RAM} {args.task} {args.physic.operator_name} {args.data.dataset_name}",  
            name=date+"_"+args.task,
            log_model=False,
            config={
                "model": "Deep Unfolding",
                "Structure": "HQS",
                "dataset": f"{args.data.dataset_name}",
            },
        )
    
    # model
    if args.task == "" or args.physic.operator_name == "":
        pl_model = um.GeneralisedUnfoldedModel(
            args.num_gpus, 
            args, 
            device=device, 
            wandb_logger=wandb_logger, 
            PnP_Forward=False
        )
        pth = os.path.join(
            args.save_checkpoint_dir, 
            f"ckpts_RAM_ud2m_generalised_{args.task}", 
        )
        str_lora="lora"
    else:
        pl_model = um.UnfoldedModel(
            physic,
            kernels,
            args.num_gpus,
            args,
            device=device,
            wandb_logger=wandb_logger,
            dphys=dinv_physic,
            PnP_Forward=False
        )
            
        if args.use_RAM:
            root_ram = f"ckpts_RAM_ud2m_{args.task}"
        else:
            root_ram = f"ckpts_ud2m_{args.task}"

        root_ckpt = os.path.join(
            args.save_checkpoint_dir, 
            f"{root_ram}"
        )
        str_lora=f"lora_{args.task}_{args.physic.operator_name}"

    os.makedirs(root_ckpt, exist_ok=True)
    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=root_ckpt,
        filename=f'{args.data.dataset_name.lower()}_{str_lora}'+'_{epoch:03d}'+f'-{args.model.rank}',
        save_top_k=2,
        every_n_epochs=10,
    )
    psnr_callback_epoch = ModelCheckpoint(
        monitor='psnr_mean',
        mode='max',
        dirpath=pth,
        filename=f'{args.data.dataset_name.lower()}_BestPSNR_{str_lora}'+'_{epoch:03d}'+f"-{args.model.rank}",
        save_top_k=1,
        every_n_epochs=1,
    )

    # --- >> profi;er for advance analysis
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    # --- >> precision 
    torch.set_float32_matmul_precision('high')
    
    # --- >> Trainer module
    trainer = pl.Trainer(accelerator="gpu", 
                         devices=args.num_gpus, 
                         strategy='ddp_find_unused_parameters_true',
                         max_epochs=args.train.num_epochs, 
                         callbacks=[checkpoint_callback_epoch, psnr_callback_epoch],
                         num_sanity_val_steps=2, 
                         profiler="simple", 
                         logger=wandb_logger, 
                         benchmark=False,
                         log_every_n_steps=5,
                         num_nodes=1,
                         limit_train_batches=args.data.number_sample_per_epoch//args.data.train_batch_size,#1000 // 32
                        #  precision="bf16",
                    )

    # --- >> Training
    if args.train.resume:
        trainer.fit(pl_model, loader_train, loader_val, ckpt_path=None)
    else:
        trainer.fit(pl_model, loader_train, loader_val)      
if __name__ == "__main__":
    main()