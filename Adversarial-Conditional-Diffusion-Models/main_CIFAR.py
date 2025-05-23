import os
import copy
import datetime
import yaml
import wandb

import torch
from torch.utils.data import DataLoader, random_split
import torch.profiler as tp
from models.load_model import load_frozen_model
from datasets.datasets import GetDatasets
from configs.args_parse import configs

import unfolded_models as um
import physics as phy
import utils as utils

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

def main():
    # logger 
    script_dir = os.path.dirname(__file__)
    # args
    date = datetime.datetime.now().strftime("%d-%m-%Y")
    config = configs()
    print(os.path.join(script_dir,"configs", config.config_file))
    with open(os.path.join(script_dir, "configs", config.config_file), 'r') as file:
        config_dict = yaml.safe_load(file)
    args = utils.dict_to_dotdict(config_dict)
    args.task = config.task
    args.physic.operator_name = config.operator_name
    if config.task == "inp":
        args.dpir.use_dpir = config.use_dpir
    args.lambda_ = config.lambda_
    args.date = date
    args.max_unfolded_iter = config.max_unfolded_iter
   
    # logger 
    script_path = script_dir.rsplit("/", 1)[0]
    log_dir = os.path.join(script_path, "Z_logs", f"Logger_CDM_{args.task}", date)
    os.makedirs(log_dir, exist_ok=True)
    logger = utils.get_loggers(log_dir, f"log_unfolded.log")
    
    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # datasets

    from diffusion.utils.data import get_data, RandomSubsetDataSampler
    from torchvision.transforms import Lambda

    data = get_data("Imagenet64", new_transforms = [Lambda(utils.image_transform),], labels = True)

    trainloader = DataLoader(
        data["train"], 
        batch_size=args.data.train_batch_size, 
        sampler = RandomSubsetDataSampler(data["train"], num_points = 32000),# 32768),
        num_workers=args.data.num_workers,
        drop_last=True,
    )
    valid_loader = DataLoader(
        data["val"], 
        batch_size=args.data.test_batch_size, 
        shuffle=False, 
        num_workers=args.data.num_workers,
        drop_last = True,
    )
    
    # physic
    dinv_physic = None
    if args.task == "deblur":
        kernels = phy.Kernels(
            operator_name=args.physic.operator_name,
            kernel_size=args.physic.kernel_size,
            device=device,
            mode = args.mode,
            std = 1.0
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
        from deepinv.physics import Inpainting, GaussianNoise
        dinv_physic = Inpainting(
            (3, 64, 64),
            mask = physic.Mask,
            device = device,
            noise_model = GaussianNoise(
                sigma= args.physic.sigma_model,
            ),
        ).to(device)
        
            
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
        from deepinv.physics import Downsampling, GaussianNoise
        dinv_physic =   Downsampling(
            filter = "bicubic",
            img_size = (3, 64, 64),
            factor = args.physic.sr.sf,
            device = device,
            noise_model = GaussianNoise(
                sigma= args.physic.sigma_model,
            ),
        ).to(device)

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
        project=f"Adversarial Conditional diffusion models {args.task}",  
        name=date+"_"+args.task+"_lambda"+str(args.lambda_)+"_"+args.physic.operator_name,
        log_model=False,
        # save_dir=args.save_checkpoint_dir + 'wandb' + date,
    )
    
    # model
    pl_model = um.UnfoldedModel(physic, kernels, args.num_gpus, args, device=device, wandb_logger=wandb_logger, max_unfoled_iter=args.max_unfolded_iter, dphys = dinv_physic)
    if args.task=="sr":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}_{args.max_unfolded_iter}", date, f"factor_{args.physic.sr.sf}")
    elif args.task=="deblur":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}_{args.max_unfolded_iter}", date, args.physic.operator_name)
    elif args.task=="inp":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}_{args.max_unfolded_iter}", date, args.physic.operator_name, f"scale_{args.physic.inp.mask_rate}")
    elif args.task=="ct":
        pth = os.path.join(args.save_checkpoint_dir, f"{args.task}_{args.max_unfolded_iter}", date,f"angles_{args.physic.ct.angles}")
    else:

        raise ValueError(f"This task ({args.task})is not implemented!!!")
         
    os.makedirs(pth, exist_ok=True)
    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=pth,
        filename='checkpoint_lora_'+ args.task +'_{epoch:03d}',
        save_top_k=1,
        every_n_epochs=1,
    )

    psnr_callback_epoch = ModelCheckpoint(
        monitor='psnr_mean',
        mode='max',
        dirpath=pth,
        filename='Bestpsnr_'+ args.task +'_{epoch:03d}',
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
    # torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision('high')
    
    # --- >> Trainer module
    trainer = pl.Trainer(accelerator="gpu", 
                         devices=args.num_gpus, 
                         strategy='ddp',
                         max_epochs=args.train.num_epochs, 
                         callbacks=[checkpoint_callback_epoch,psnr_callback_epoch],
                         num_sanity_val_steps=2, 
                         profiler="simple", 
                         logger=wandb_logger, 
                         benchmark=False,
                         log_every_n_steps=100,
                        #  precision="bf16",
                         num_nodes=1,
                    )

    # --- >> Training
    # with tp.profile() as prof:
    ## Print parameters:
    # for k, v in pl_model.named_parameters():
    #     if v.requires_grad:
    #         print(k, v.shape, f'requires_grad={v.requires_grad}')
        # print(k, v.shape, f'requires_grad={v.requires_grad}')
    # exit()
    if args.train.resume:
        trainer.fit(pl_model, trainloader, valid_loader,
                ckpt_path=args.save_checkpoint_dir + args.task + f'/checkpoint-epoch-{args.train.resume_epoch}.ckpt')
    else:
        trainer.fit(pl_model, trainloader, valid_loader)  
    
    # prof.key_averages().table(sort_by="cpu_time_total")
    
if __name__ == "__main__":
    main()