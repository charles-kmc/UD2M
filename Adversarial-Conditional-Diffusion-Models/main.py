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


def extract_image_paths(root_dir):
    image_paths = []
    labels = []
    
    # Traverse through the root directory
    df = pd.DataFrame(columns=["img_path"])
    s_dir = "/users/cmk2000/sharedscratch/Datasets/LSUN/infos"
    os.makedirs(s_dir, exist_ok=True)
    for root, dirs, files in os.walk(root_dir):
        d = []
        for file in files:
            # Assuming images are jpg, jpeg, png. You can extend it to other formats.
            if file.endswith(('.webp')):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)  # Label is the folder name
                d.append(image_path)
                new_row = pd.DataFrame({"img_path": d})
                df = pd.concat([df, new_row], ignore_index=True)
                
        df.to_csv(os.path.join(s_dir,"lsun_bedroom_infos.csv"), index=False)

def main():
    # logger 
    script_dir = os.path.dirname(__file__)
    # args
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    config = configs()
    with open(os.path.join(script_dir, "configs", config.config_file), 'r') as file:
        config_dict = yaml.safe_load(file)
    args = utils.dict_to_dotdict(config_dict)
    args.task = config.task
    args.data.dataset_name = config.dataset
    args.physic.operator_name = config.operator_name
    if config.task == "inp":
        args.dpir.use_dpir = config.use_dpir
    args.lambda_ = config.lambda_
    args.date = date
    args.max_unfolded_iter = 3
    
    # logger 
    script_path = script_dir.rsplit("/", 1)[0]
    log_dir = os.path.join(script_path, "Z_logs", f"Logger_CDM_{args.task}", date)
    os.makedirs(log_dir, exist_ok=True)
    logger = utils.get_loggers(log_dir, f"log_unfolded.log")
    
    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # datasets
    if args.data.dataset_name=="FFHQ":
        args.data.dataset_path =  f"{args.data.root}/ffhq_dataset"
    elif args.data.dataset_name=="CT":
        args.data.dataset_path = f"{args.data.root}/training_CT_data/trainingset_true_tiff"
    elif args.data.dataset_name=="LSUN":
        args.data.dataset_path =  f"{args.data.root}/LSUN"
    elif args.data.dataset_name=="ImageNet":
        args.data.dataset_path =  ""
        raise ValueError("Not implemented !!")
        
    
    # if True:
    #     import pandas as pd
    #     print(args.data.number_sample_per_epoch)
    #     extract_image_paths("/users/cmk2000/sharedscratch/Datasets/LSUN/train")
    # sderf
    
    
    # datasets
    prop = 0.2
    prop2 = 0.001
    if args.data.dataset_name=="LSUN":
        train_dataset = GetDatasets(
            args.data.dataset_path, 
            args.data.im_size, 
            dataset_name=args.data.dataset_name,
            type_="train",
        )
        valid_dataset = GetDatasets(
            args.data.dataset_path, 
            args.data.im_size, 
            dataset_name=args.data.dataset_name,
            type_="val",
        )
    else:
        datasets = GetDatasets(
            args.data.dataset_path, 
            args.data.im_size, 
            dataset_name=args.data.dataset_name,
        )
        train_size = int(prop * len(datasets))
        valid_size = len(datasets) - train_size
        val = int(prop2 * valid_size)
        valid_size = valid_size - val
        train_dataset, valid_dataset, _ = random_split(
            datasets, 
            [train_size, val, valid_size]
        )
    sampler = RandomSampler(train_dataset, num_samples=args.data.number_sample_per_epoch, replacement=False) if args.data.is_random_sampler else None
    
    trainloader = DataLoader(
        train_dataset, 
        batch_size = args.data.train_batch_size, 
        shuffle = args.data.shuffle if not args.data.is_random_sampler else False, 
        num_workers = args.data.num_workers,
        sampler = sampler
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.data.test_batch_size, 
        shuffle=args.data.shuffle, 
        num_workers=args.data.num_workers
    )
    
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
            scale_image=args.physic.transform_y,
        )
    elif args.task == "inp":
        physic = phy.Inpainting(
            mask_rate=args.physic.inp.mask_rate,
            im_size=args.data.im_size,
            operator_name=args.physic.operator_name,
            box_proportion = args.physic.inp.box_proportion,
            index_ii = args.physic.inp.index_ii,
            index_jj = args.physic.inp.index_jj,
            device=device,
            sigma_model=args.physic.sigma_model,
            scale_image=args.physic.transform_y,
            mode=args.mode,
        )
        kernels = phy.Kernels(
            operator_name=args.physic.operator_name,
            kernel_size=args.physic.kernel_size,
            device=device,
            mode = args.mode
        )
    elif args.task == "sr":
        physic = phy.SuperResolution(
            im_size=args.data.im_size,
            sigma_model=args.physic.sigma_model,
            device=device,
            scale_image=args.physic.transform_y,
            mode = args.mode,
        )
        kernels = phy.Kernels(
            operator_name=args.physic.operator_name,
            kernel_size=args.physic.kernel_size,
            device=device,
            mode = args.mode
        )

    elif args.task == "ct":
        physic = phy.CT(
            im_size=args.data.im_size,
            angles=args.physic.ct.angles,
            sigma_model=args.physic.sigma_model,
            device=device,
            scale_image=args.physic.transform_y,
            mode = args.mode,
        )
        kernels = phy.Kernels(
            operator_name=args.physic.operator_name,
            kernel_size=args.physic.kernel_size,
            device=device,
            mode = args.mode
        )
    else:
        raise ValueError(f"This task ({args.task})is not implemented!!!")
    
    os.makedirs(os.path.join(args.save_checkpoint_dir, 'wandb', date), exist_ok=True)
    wandb_logger = WandbLogger(
        project=f"Adversarial Conditional diffusion models {args.task} {args.physic.operator_name}",  
        name=date+"_"+args.task+"_lambda"+str(args.lambda_)+"_"+args.physic.operator_name+f"_{args.data.dataset_name}",
        log_model=False,
        # save_dir=args.save_checkpoint_dir + 'wandb' + date,
    )
    
    # model
    pl_model = um.UnfoldedModel(physic, kernels, args.num_gpus, args, wandb_logger=wandb_logger)
    if args.task=="sr":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}", date, f"factor_{args.physic.sr.sf}")
    elif args.task=="deblur":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}", date, args.physic.operator_name)
    elif args.task=="inp":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}", date, args.physic.operator_name, f"scale_{args.physic.inp.mask_rate}")
    elif args.task=="ct":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}", date,f"angles_{args.physic.ct.angles}")
    else:
        raise ValueError(f"This task ({args.task})is not implemented!!!")
         
    os.makedirs(pth, exist_ok=True)
    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=pth,
        filename=f'{args.data.dataset_name.lower()}_lora_{args.task}'+'_{epoch:03d}',
        save_top_k=40,
        every_n_epochs=10,
    )

    # --- >> profi;er for advance analysis
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    # --- >> precision 
    # torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision('high')
    
    # --- >> Trainer module
    trainer = pl.Trainer(accelerator="gpu", 
                         devices=args.num_gpus, 
                         strategy='ddp',
                         max_epochs=args.train.num_epochs, 
                         callbacks=[checkpoint_callback_epoch],
                         num_sanity_val_steps=2, 
                         profiler="simple", 
                         logger=wandb_logger, 
                         benchmark=False,
                         log_every_n_steps=5,
                         num_nodes=1,
                         limit_train_batches=args.data.number_sample_per_epoch//args.data.train_batch_size#1000 // 32
                        #  precision="bf16",
                    )

    # --- >> Training
    # with tp.profile() as prof:
    if args.train.resume:
        trainer.fit(pl_model, trainloader, valid_loader,
                ckpt_path=args.save_checkpoint_dir + args.task + f'/checkpoint-epoch-{args.train.resume_epoch}.ckpt')
    else:
        trainer.fit(pl_model, trainloader, valid_loader)  
    
    # prof.key_averages().table(sort_by="cpu_time_total")
    
if __name__ == "__main__":
    main()