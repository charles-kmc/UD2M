import os
import copy
import time
import yaml
import numpy as np
import pandas as pd
import blobfile as bf
import matplotlib.pyplot as plt
import datetime
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from datasets.datasets import GetDatasets
import models as models
import utils as utils
import unfolded_models as ums
import physics as phy
import runners as runners
from configs.args_parse import configs
from metrics.metrics import Metrics
from datasets.get_datasets import get_dataset, get_dataloader

max_images = 30
max_iter = 1
LORA = True
ZETA = [0.9]

def main(
    solver_type:str, 
    eta:float=0.0,
    zeta:float=0.9,
    **kwargs,
):
    with torch.no_grad():
        # args
        mode = "inference"
        
        script_dir = os.path.dirname(__file__)
        # args
        config = configs(mode=mode)
        
        with open(os.path.join(script_dir, "configs", config.config_file), 'r') as file:
            config_dict = yaml.safe_load(file)
        args = utils.dict_to_dotdict(config_dict)
        args.task = config.task
        args.mode = mode
        args.data.dataset_name = config.dataset_name
        args.physic.operator_name = config.operator_name
        args.physic.sigma_model = config.sigma_model
        args.max_unfolded_iter = config.max_unfolded_iter
        MAX_UNFOLDED_ITER = [args.max_unfolded_iter,]
        num_timesteps = config.num_timesteps
        steps = num_timesteps 
        
        if config.task == "inp":
            args.dpir.use_dpir = config.use_dpir
        args.lambda_ = config.lambda_
        args.learned_lambda_sig = config.learned_lambda_sig
        args.lora = LORA
        args.evaluation.coverage = False
        args.ddpm_param = ddpm_param
        args.solver_type = solver_type
        args.eta = eta
        args.lora_checkpoint_dir = config.ckpt_dir
        args.lora_checkpoint_name = config.ckpt_name
        max_images = config.max_images
        args.physic.kernel_size = config.kernel_size
        
        # parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        args.device = device
        args.zeta = zeta
        
        # 
        for max_unfolded_iter in MAX_UNFOLDED_ITER:
            args.max_unfolded_iter = max_unfolded_iter
            args.save_wandb_img = False                
                
            if args.dpir.use_dpir and args.task!="inp":
                init = f"init_model_{args.dpir.model_name}"
            else:
                init = f"init_xty_{args.init_xt_y}"
            
            if args.use_wandb:
                import wandb
                formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                wandb.init(
                        # set the wandb project where this run will be logged
                        project=f"Sampling Unfolded Conditional Diffusion Models {args.task}_{args.physic.operator_name}",
                        config={
                            "model": "Deep Unfolding",
                            "Structure": "UD2M",
                            "dataset": f"{args.data.dataset_name}",
                        },
                        name = f"{formatted_time}_task_{args.task}_rank_{args.model.rank}",
                    )
            # data
            if args.data.dataset_name=="ImageNet":
                # -- validation dataset
                args.data.dataset_path =  f"{args.data.root}/{args.data.dataset_name}"
                datasets = GetDatasets(args.data.dataset_path, im_size=args.im_size, type_="val", dataset_name=args.data.dataset_name)
                loader = DataLoader(datasets, batch_size=1, shuffle=False)
            else:                   
                dataset_val = get_dataset(
                    name=args.data.dataset_name, 
                    root_dataset=args.data.root, 
                    im_size=args.data.im_size, 
                    subset_data="val",
                )
                loader = get_dataloader(dataset_val, args=args, subset_data="val")  
            
            # -- model
            model = models.adapter_lora_model(args)  
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
            
            # Diffusion noise
            diffusion_scheduler = ums.DiffusionScheduler(device=device,noise_schedule_type=args.noise_schedule_type)
            
            # physic
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
            elif args.task == "phase_retrieval":
                physic_configs = {
                    "oversampling_ratio": args.physic.phase_retrieval.oversampling_ratio,
                }
            elif args.task == "jpeg":
                physic_configs = {
                    "qf": args.physic.jpeg.qf,
                }
        
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
            if args.task in ["deblur", "sr", "inp", "jpeg"]:
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
                    (3,256, 256),
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
            
            # HQS module
            denoising_timestep = ums.GetDenoisingTimestep(diffusion_scheduler, device)
            hqs_module = ums.HQS_models(
                model,
                physic, 
                diffusion_scheduler, 
                denoising_timestep,
                args,
                device=device
            )
            try:
                hqs_module.load_state_dict(trainable_state_dict["HQS_state_dict"], strict=False)
                state_dict_ram = trainable_state_dict["RAM_state_dict"]
            except Exception as e:
                hqs_module.load_state_dict(trainable_state_dict["state_dict"], strict=False)
                state_dict_ram = None
            hqs_module = hqs_module.to(device)

            # sampler               
            diffsampler = runners.Conditional_sampler(
                hqs_module, 
                physic, 
                diffusion_scheduler, 
                device, 
                args,
                dphys=dinv_physic,
                max_unfolded_iter=max_unfolded_iter,
                state_dict_RAM=state_dict_ram
            )

            # Metrics
            metrics = Metrics(device=device)
            
            # directories                
            root_dir_results= os.path.join(args.save_dir, 
                            args.data.dataset_name, 
                            f"{args.task}_operator_{args.physic.operator_name}", 
                            solver_type
            )
            
            if args.evaluation.metrics:
                save_metric_path = os.path.join(root_dir_results, "metrics_results")
                os.makedirs(save_metric_path, exist_ok=True)
                
            if args.evaluation.save_images:
                ref_path = os.path.join(root_dir_results, "ref")
                mmse_path = os.path.join(root_dir_results, "mmse")
                metric_path = os.path.join(root_dir_results, "metrics")
                var_path = os.path.join(root_dir_results, "var")
                y_path = os.path.join(root_dir_results, "y")
                last_path = os.path.join(root_dir_results, "last")
                for dir_ in [ref_path, mmse_path, y_path, last_path, var_path,metric_path]:
                    os.makedirs(dir_, exist_ok=True)
            
            # Evaluations
            s_psnr,s_mse,s_ssim,s_lpips,count = 0,0,0,0,0 
            
            # Blur kernel 
            if args.task == "deblur" or args.task=="sr":    
                blur = kernels.get_blur(seed=1234)
            
            # loop over loader 
            for ii, im in enumerate(loader):
                if isinstance(im, list):
                    if isinstance(im[1], dict) and "y_label" in im[1].keys():
                        im, y_label = im[0], im[1]["y_label"]
                        y_label = y_label.to(device)
                else:
                    y_label = None
                if ii*im.shape[0]>max_images:continue
                
                # True image
                im_name = f"{(ii):05d}"
                im = im.to(device)
                x = utils.inverse_image_transform(im)
                x_true = None
                
                # observation y 
                if args.task == "deblur":
                    y = physic.y(im, blur)
                elif args.task=="sr":
                    y = physic.y(im, blur, args.physic.sr.sf)
                elif args.task in ["inp", "jpeg"]:
                    y = physic.y(im)
                        
                #  main loop  
                for it in range(max_iter):
                    start_time = time.time()
                    out = diffsampler.sampler(
                        y, 
                        im_name, 
                        num_timesteps = num_timesteps, 
                        eta=eta, 
                        zeta=zeta,
                        x_true = x_true,
                        lambda_ = args.lambda_,
                        y_label = y_label,
                    )

                    # collapsed time
                    collapsed_time = time.time() - start_time           
                    # sample
                    if it == 0:
                        posterior = utils.welford(out["xstart_pred"])
                    if it > 0:
                        # posterior
                        posterior.update(out["xstart_pred"])
                        if args.use_wandb and ii%10==0 and args.evaluation.coverage:
                            psnr_val = metrics.psnr_function(out["xstart_pred"], x)
                            wandb.log({f"sample psnr": psnr_val})
                # posterior mean - mmse
                if max_iter>1:
                    X_posterior_mean = posterior.get_mean()
                    X_posterior_var = posterior.get_var()
                else:
                    X_posterior_mean = out["xstart_pred"]
                    X_posterior_var = out["xstart_pred"]
                utils.delete([posterior])
                    
                last_sample = data[-1].to(device) if args.evaluation.coverage else out["xstart_pred"]
                
                # sliced wasserstein distance
                if args.evaluation.save_images:   
                    for ll, (x_t, xp_t, y_t, v_t) in enumerate(zip(x, X_posterior_mean, y, X_posterior_var)):
                        im_name = f"{(ii*x.shape[0] + ll):05d}"
                        utils.save_images(ref_path, x_t, im_name)
                        utils.save_images(mmse_path, xp_t, im_name)
                        std = v_t
                        err = torch.abs(xp_t-x_t)
                        norm = max(std.max(), err.max())
                        utils.save_images(var_path, err /norm, im_name+"_residual")
                        utils.save_images(y_path, utils.inverse_image_transform(y_t), im_name)                            
                        # eval - mmse
                        psnr_temp = metrics.psnr_function(xp_t, x_t)
                        mse_temp = metrics.mse_function(xp_t, x_t)
                        ssim_temp = metrics.ssim_function(xp_t, x_t)
                        lpips_temp = metrics.lpips_function(xp_t, x_t)
                        # eval - last sample
                        psnr_temp_last = metrics.psnr_function(last_sample[ll], x_t)
                        mse_temp_last = metrics.mse_function(last_sample[ll], x_t)
                        ssim_temp_last = metrics.ssim_function(last_sample[ll], x_t)
                        lpips_temp_last = metrics.lpips_function(last_sample[ll], x_t)
                        matrics_pd = pd.DataFrame({
                            "name": [im_name],
                            "psnr": [psnr_temp],
                            "mse": [mse_temp],
                            "ssim": [ssim_temp],
                            "lpips": [lpips_temp],
                            "time": [collapsed_time],
                            "psnr last": [psnr_temp_last],
                            "mse last": [mse_temp_last],
                            "ssim last": [ssim_temp_last],
                            "lpips last": [lpips_temp_last],
                        })
                        save_dir_metric = os.path.join(save_metric_path, 'metrics_results.csv')
                        matrics_pd.to_csv(save_dir_metric, mode='a', header=not os.path.exists(save_dir_metric))
                        s_lpips += lpips_temp
                        s_psnr += psnr_temp
                        s_mse += mse_temp
                        s_ssim += ssim_temp
                        count += 1
            wandb.finish()   

if __name__ == "__main__":
    for solver in SOLVER_TYPES:
            for eta in etas:
                kwargs = {
                    "solver_type": "ddim",
                    "eta": 0.8,
                }
                main(**kwargs)
