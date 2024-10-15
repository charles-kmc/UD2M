import os
import copy
import time
import numpy as np
import pandas as pd
import datetime
import wandb
import torch
from torch.utils.data import DataLoader

from unfolded_models.unfolded_model import (
    HQS_models, 
    physics_models, 
    DiffusionScheduler,
    GetDenoisingTimestep,
    get_rgb_from_tensor,
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
from utils.utils import ( 
    inverse_image_transform, 
    image_transform,
    DotDict,
    welford,
    save_images,
)
from metrics.coverage.coverage_function import coverage_eval
from metrics.sliced_wasserstein_distance.evaluate_swd import eval_swd
from metrics.metrics import Metrics
from metrics.fid_batch.fid_evaluation import Fid_evatuation
import deepinv



def main():
    # args
    args = args_unfolded()
    
    # parameters
    LORA = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu") #deepinv.utils.get_freer_gpu()
    # print("00000000", device,  torch.cuda.is_available())
    # print(torch.cuda.is_available())  # Should return True if CUDA is available
    
    # print(torch.cuda.current_device())  # Should return the index of the current GPU
    # print(torch.cuda.device_count())  # Should return the number of GPUs available
    # print(torch.cuda.get_device_name(0))
    
    # data
    dir_test = "/users/cmk2000/sharedscratch/Datasets/testsets/imageffhq"
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
    args.date = "14-10-2024"
    args.epoch = 30
    args.eval_date = datetime.datetime.now().strftime("%d-%m-%Y")
    args.save_wandb_img = False
    #args.dpir.model_name = 'drunet_color', #"ircnn_color, drunet_color"
    
    if LORA:
        LoRa_model(
            model, 
            target_layer = ["qkv", "proj_out"], 
            device=device, 
            rank = args.rank
        )
        load_trainable_params(model, args.epoch, args)
        
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
    max_unfolded_iter = 3
    diffsampler = DDIM_SAMPLER(
        hqs_model, 
        physics, 
        diffusion, 
        device, 
        args,
        max_unfolded_iter = max_unfolded_iter
    )
        
    # parameters
    max_iter = 150
    num_timesteps = 100
    eta = 0.0
    zeta = 0.3
    args.zeta = zeta
    args.eta = eta
    args.max_unfolded_iter = max_unfolded_iter
    
    # Metrics
    metrics = Metrics(device=device)
    
    # monitor sample for the swd
    dict_metrics = {
        "name":[], 
        "psnr":[], 
        "mse":[], 
        "ssim":[], 
        "lpips":[], 
        "time":[]
    }

    dic_swd = {
        "Total images": [], 
        "swd mmse": [], 
        "swd last sample": [], 
        "psnr mean": [], 
        "mse mean": [], 
        "lpips mean": [], 
        "ssim mean": []
    }
    samples_ref_y = []
    samples_ref_mmse = []
    samples_ref_last = []
    
    # directories
    root_dir_results = os.path.join(args.path_save, args.task, "Results", args.eval_date, f"Max_iter_{max_iter}", f"timesteps_{num_timesteps}", f"Epoch_{args.epoch}")
    if args.evaluation.coverage or args.evalution.metrics:
        save_metric_path = os.path.join(root_dir_results, "metrics_results")
        os.makedirs(save_metric_path, exist_ok=True)
        
    if args.evaluation.fid:
        ref_path = os.path.join(root_dir_results, "ref")
        mmse_path = os.path.join(root_dir_results, "mmse")
        y_path = os.path.join(root_dir_results, "y")
        last_path = os.path.join(root_dir_results, "last")
        for dir_ in [ref_path, mmse_path, y_path, last_path]:
            os.makedirs(dir_, exist_ok=True)
        
    
    # loop over testset   
    for ii, im in enumerate(testset):
        # obs
        im = im.to(device)
        y = physics.y(im)
        x = inverse_image_transform(im)
        #sampling
        im_name = f"_{(ii):05d}"
        
        # coverage 
        if args.evaluation.coverage:
            samples = []
     
        
        for it in range(max_iter):
            start_time = time.time()
            out = diffsampler.sampler(
                im,
                y, 
                im_name, 
                num_timesteps = num_timesteps, 
                eta=eta, 
                zeta=zeta
            )
            
            # collapsed time
            collapsed_time = start_time - time.time()
            
            # sample
            xi = out["xstart_pred"]
            zi = out["auxiliary"]
            if it ==1:
                posterior = welford(xi)
                
            if args.evaluation.coverage and it>1:
                samples.append(xi)
                posterior.update(xi)
                
                if args.use_wandb and ii%10==0:
                    psnr_temp = metrics.psnr_function(xi, x)
                    wandb.log({f"sample psnr {im_name}": psnr_temp})
        # posterior mean - mmse
        X_posterior_mean = posterior.get_mean()
                
        if args.use_wandb and ii%10 == 0:
            im_wdb = wandb.Image(
                get_rgb_from_tensor(x), 
                caption=f"true image", 
                file_type="png"
            )
            x0_wdb = wandb.Image(
                get_rgb_from_tensor(xi), 
                caption=f"x_est psnr/mse ({metrics.psnr_function(xi, x):.2f},{metrics.lpips_function(xi, x):.2f}))", 
                file_type="png"
            )
            xmmse_wdb = wandb.Image(
                get_rgb_from_tensor(X_posterior_mean), 
                caption=f"x_mmse psnr/lpips ({metrics.psnr_function(X_posterior_mean, x):.2f},{metrics.lpips_function(X_posterior_mean, x):.2f})", 
                file_type="png"
            )
            z0_wdb = wandb.Image(
                get_rgb_from_tensor(zi), 
                caption=f"z_est psnr/mse ({metrics.psnr_function(zi, x):.2f},{metrics.lpips_function(zi, x):.2f})", 
                file_type="png"
            )
            
            y_wdb = wandb.Image(
                get_rgb_from_tensor(inverse_image_transform(y)), 
                caption=f"observation", 
                file_type="png"
            )
            xy = [im_wdb, y_wdb, x0_wdb, z0_wdb, xmmse_wdb]
            wandb.log({"blurred and predict":xy})
            
            # progression
            image_wdb = wandb.Image(out["progress_img"], caption=f"progress sequence for im {im_name}", file_type="png")
            wandb.log({"pregressive":image_wdb})
        
        # coverage
        if args.evaluation.coverage:
            op_coverage = DotDict()
            op_coverage.samples = []; op_coverage.inside_interval = []; op_coverage.num_dist_greater = []
            op_coverage.im_num = max_iter*[ii]; op_coverage.truexdist_list = []; op_coverage.quantilelist_list = []
            op_coverage.save_dir = save_metric_path
            op_coverage.samples = samples
            op_coverage.post_mean = X_posterior_mean.detach().cpu()
            op_coverage.prob = np.linspace(0.05,0.99,50).round(2) #[0.8, 0.85, 0.875, 0.90, 0.95, 0.975, 0.99, 0.999]
            coverage_eval(x.detach().cpu(), op_coverage)
        
        # sliced wasserstein distance
        if args.evaluation.swd:    
            ref = x.detach().cpu().squeeze()
            blurry = y.detach().cpu().squeeze()
            sample = samples[-1].detach().cpu().squeeze()
            mmse = X_posterior_mean.detach().cpu().squeeze()
            ref_y = torch.hstack([ref, blurry]).permute(0,2,1)
            ref_mmse = torch.hstack([ref, mmse]).permute(0,2,1)
            ref_last = torch.hstack([sample, blurry]).permute(0,2,1)
            
            # save images for fid and cmmd evaluation
            save_images(ref_path, im, im_name)
            save_images(mmse_path, X_posterior_mean, im_name)
            save_images(y_path, y, im_name)
            save_images(last_path, samples[-1], im_name)
            
            # collect data
            samples_ref_y.append(ref_y)
            samples_ref_mmse.append(ref_mmse)
            samples_ref_last.append(ref_last)
            
            # metrics
            dict_metrics["name"].append(im_name)
            dict_metrics["psnr"].append(metrics.psnr_function(X_posterior_mean, x))
            dict_metrics["mse"].append(metrics.mse_function(X_posterior_mean, x))
            dict_metrics["ssim"].append(metrics.ssim_function(X_posterior_mean, x))
            dict_metrics["lpips"].append(metrics.lpips_function(X_posterior_mean, x))
            dict_metrics["time"].append(collapsed_time)
            
            # save
            matrics_pd = pd.DataFrame(dict_metrics)
            save_dir_metric = os.path.join(save_metric_path, 'metrics_results.csv')
            matrics_pd.to_csv(save_dir_metric, mode='a', header=not os.path.exists(save_dir_metric))
            
        # sliced wasserstein distance              
        if args.evaluation.swd:    
            samples_ref = torch.stack(samples_ref_y)
            samples_mmse = torch.stack(samples_ref_mmse)
            samples_last = torch.stack(samples_ref_last)

            
            if ii % args.evaluation.skipper == 0 and ii >= args.evaluation.skipper:
                swd_ref_mmse = np.round(eval_swd(samples_ref, samples_mmse).item(),2)
                swd_ref_last = np.round(eval_swd(samples_ref, samples_last).item(),2)

                dic_swd["Total images"].append(ii+1)
                dic_swd["swd mmse"].append(swd_ref_mmse)
                dic_swd["swd last sample"].append(swd_ref_last)
                
                dic_swd["psnr mean"].append(np.mean(np.array(dict_metrics["psnr"])))
                dic_swd["lpips mean"].append(np.mean(np.array(dict_metrics["lpips"])))
                dic_swd["ssim mean"].append(np.mean(np.array(dict_metrics["ssim"])))
                dic_swd["mse mean"].append(np.mean(np.array(dict_metrics["mse"])))
                
                # save results as pandas dataframe
                data_pd = pd.DataFrame(dic_swd)
                save_dir_swd = os.path.join(save_metric_path, 'swd_results.csv')
                data_pd.to_csv(save_dir_swd, mode='a', header=not os.path.exists(save_dir_swd))
    
    # FiD
    if args.evaluation.fid:
        fid_val = Fid_evatuation(root_dir_results, device)
        dic_fid = {"fid":fid_val}
        fid_pd = pd.DataFrame(dic_fid, index = [0])
        save_dir_fid = os.path.join(save_metric_path, 'fid_results.csv')
        fid_pd.to_csv(save_dir_fid, mode='a', header=not os.path.exists(save_dir_fid))

if __name__ == "__main__":
    main()