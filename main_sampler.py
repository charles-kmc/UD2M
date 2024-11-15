import os
import copy
import time
import sys
import numpy as np
import pandas as pd
import psutil
import datetime
import flags

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
    delete,
)
from metrics.coverage.coverage_function import coverage_eval
from metrics.sliced_wasserstein_distance.evaluate_swd import eval_swd
from metrics.metrics import Metrics
from metrics.fid_batch.fid_evaluation import Fid_evatuation


SOLVER_TYPES =  ["ddpm"]
LORA = True
MAX_UNFOLDED_ITER = [3]
NUM_TIMESTEPS = [20, 50, 100, 150, 200, 250, 350, 500, 600, 700, 800, 900, 1000]
ETA = [0.2] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
T_AUG = 4
ZETA = [0.0]#6, 0.7, 0.8, 0.9, 1]#np.arange(0.6, 1, 0.1)

date_eval = datetime.datetime.now().strftime("%d-%m-%Y")

def main(num_timesteps, solver_type, ddpm_param = 1):
    with torch.no_grad():
        # args
        args = args_unfolded()
        
        args.ddpm_param = ddpm_param
        # parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        max_iter = 150 if args.evaluation.coverage else 1
        args.solver_type = solver_type
        for eta in ETA:
            args.eta = eta
            for idx in range(len(ZETA)):
                zeta = ZETA[idx]
                args.zeta = zeta
                for max_unfolded_iter in MAX_UNFOLDED_ITER:
                    args.max_unfolded_iter = max_unfolded_iter

                    args.mode = "eval"
                    args.date = "22-10-2024"
                    args.epoch = 600
                    args.eval_date = date_eval
                    args.save_wandb_img = False
                    #args.dpir.model_name = 'drunet_color', #"ircnn_color, drunet_color"
                    
                    if args.dpir.use_dpir:
                        init = f"init_model_{args.dpir.model_name}"
                    else:
                        init = f"init_xty_{args.init_xt_y}"
                    
                    if args.use_wandb:
                        import wandb
                        formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        dir_w = "/users/cmk2000/sharedscratch/Results/wandb"
                        os.makedirs(dir_w, exist_ok=True)
                        if solver_type == "ddim":
                            solver_params_title = f"eta{args.eta}_zeta_{args.zeta}"
                        elif solver_type == "ddpm":
                            solver_params_title = f"param_{args.ddpm_param}"
                        else:
                            raise ValueError(f"Solver {solver_type} do not exists.")
                        wandb.init(
                                # set the wandb project where this run will be logged
                                project="Sampling Unfolded Conditional Diffusion Models",
                                dir = dir_w,
                                config={
                                    "model": "Deep Unfolding",
                                    "Structure": "HQS",
                                    "dataset": "FFHQ - 256",
                                    "dir": dir_w,
                                    "pertub":args.pertub,
                                },
                                name = f"{formatted_time}_{solver_type}_num_steps_{num_timesteps}_unfolded_iter_{args.max_unfolded_iter}_{init}_task_{args.task}_rank_{args.rank}_{solver_params_title}_epoch_{args.epoch}_T_AUG_{T_AUG}",
                            )
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
                        transform_y= args.physic.transform_y,
                        random_blur=args.physic.random_blur,
                        mode = "test",
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
                        args,
                        max_unfolded_iter = max_unfolded_iter
                    )

                    # Metrics
                    metrics = Metrics(device=device)
                    
                    # monitor sample for the swd
                    samples_ref_y = []
                    samples_mmse_y = []
                    samples_last_y = []
                    

                    # PSNR paths 
                    psnr_paths = []
                    
                    # directories
                    if solver_type == "ddim":
                        param_name_folder = os.path.join(f"zeta_{zeta}", f"eta_{eta}")
                    elif solver_type == "ddpm":
                        param_name_folder = f"ddpm_param_{args.ddpm_param}"
                    else:
                        raise ValueError(f"Solver {solver_type} do not exists.")
                    root_dir_results = os.path.join(args.path_save, args.task, "Results", args.eval_date, solver_type, f"Max_iter_{max_iter}", f"unroll_iter_{args.max_unfolded_iter}", f"timesteps_{num_timesteps}", f"Epoch_{args.epoch}", f"T_AUG_{T_AUG}", param_name_folder)
                    if args.evaluation.coverage or args.evaluation.metrics:
                        save_metric_path = os.path.join(root_dir_results, "metrics_results")
                        os.makedirs(save_metric_path, exist_ok=True)
                        
                    if args.evaluation.fid:
                        ref_path = os.path.join(root_dir_results, "ref")
                        mmse_path = os.path.join(root_dir_results, "mmse")
                        y_path = os.path.join(root_dir_results, "y")
                        last_path = os.path.join(root_dir_results, "last")
                        for dir_ in [ref_path, mmse_path, y_path, last_path]:
                            os.makedirs(dir_, exist_ok=True)
                    
                    s_psnr = 0
                    s_mse = 0
                    s_ssim = 0
                    s_lpips = 0
                    count = 0    
                    # loop over testset 
                    for ii, im in enumerate(testset):
                        if ii > 200:
                            continue
                        
                        # obs
                        im = im.to(device)
                        y = physics.y(im)
                        x = inverse_image_transform(im)
                        #sampling
                        im_name = f"{(ii):05d}"
                        
                        # coverage 
                        if args.evaluation.coverage:
                            shape = x.squeeze().shape
                            shapes=(max_iter-1,) + shape
                            data = torch.zeros(shapes, dtype = x.dtype)
                            
                        for it in range(max_iter):
                            start_time = time.time()
                            out = diffsampler.sampler(
                                y, 
                                im_name, 
                                num_timesteps = num_timesteps, 
                                eta=eta, 
                                zeta=zeta,
                                x_true = x
                            )
                            
                            psnr_paths.append(out["psnrs"])
                            
                            process = psutil.Process(os.getpid()) 
                            memo1 = process.memory_info().rss / 1024 ** 2
                            wandb.log({
                                        "M1":memo1,
                                    })
                            
                            # collapsed time
                            collapsed_time = time.time() - start_time
                                        
                            # sample
                            if it ==0:
                                posterior = welford(out["xstart_pred"])
                            
                            if it>0:
                                # posterior
                                posterior.update(out["xstart_pred"])
                                # coverage 
                                if args.evaluation.coverage:
                                    data[it-1] = out["xstart_pred"].detach().squeeze()
                                
                                if args.use_wandb and ii%10==0 and args.evaluation.coverage:
                                    psnr_val = metrics.psnr_function(out["xstart_pred"], x)
                                    wandb.log({
                                        f"sample psnr {im_name}": psnr_val,
                                    })

                        # posterior mean - mmse
                        X_posterior_mean = posterior.get_mean()
                        delete([posterior])
                        
                        # psnr 
                        if args.use_wandb:
                            psnr_val = metrics.psnr_function(out["xstart_pred"], x)
                            wandb.log({
                                f"psnrs": psnr_val,
                            }) 
                            
                        if args.use_wandb and ii%20 == 0:
                            im_wdb = wandb.Image(
                                get_rgb_from_tensor(x), 
                                caption=f"true image", 
                                file_type="png"
                            )
                            x0_wdb = wandb.Image(
                                get_rgb_from_tensor(out["xstart_pred"]), 
                                caption=f"x_est psnr/mse ({metrics.psnr_function(out["xstart_pred"], x):.2f},{metrics.lpips_function(out["xstart_pred"], x):.2f}))", 
                                file_type="png"
                            )
                            xmmse_wdb = wandb.Image(
                                get_rgb_from_tensor(X_posterior_mean), 
                                caption=f"x_mmse psnr/lpips ({metrics.psnr_function(X_posterior_mean, x):.2f},{metrics.lpips_function(X_posterior_mean, x):.2f})", 
                                file_type="png"
                            )
                            z0_wdb = wandb.Image(
                                get_rgb_from_tensor(out["aux"]), 
                                caption=f"z_est psnr/mse ({metrics.psnr_function(out["aux"], x):.2f},{metrics.lpips_function(out["aux"], x):.2f})", 
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
                            delete([image_wdb, xy, xmmse_wdb, z0_wdb, y_wdb, x0_wdb, im_wdb])
                        
                        # coverage
                        if args.evaluation.coverage:
                            op_coverage = DotDict()
                            op_coverage.inside_interval = []; op_coverage.num_dist_greater = []
                            op_coverage.im_num = [ii]; op_coverage.truexdist_list = []; op_coverage.quantilelist_list = []
                            op_coverage.save_dir = save_metric_path
                            op_coverage.data = data
                            op_coverage.post_mean = X_posterior_mean.detach().cpu()
                            op_coverage.prob = np.arange(0.101, 1, 0.003) #[0.8, 0.85, 0.875, 0.90, 0.95, 0.975, 0.99, 0.999]
                            coverage_eval(x.detach().cpu(), op_coverage)
                        
                        last_sample = data[-1] if args.evaluation.coverage else out["xstart_pred"]
                        delete([data, op_coverage]) if args.evaluation.coverage else None
                        
                        # sliced wasserstein distance
                        if args.evaluation.swd:    
                            # save images for fid and cmmd evaluation
                            save_images(ref_path, x, im_name)
                            save_images(mmse_path, X_posterior_mean, im_name)
                            save_images(y_path, inverse_image_transform(y), im_name)
                            save_images(last_path, last_sample, im_name)
                            
                            # # collect data
                            # samples_ref_y.append(torch.hstack([
                            #     x.detach().cpu().squeeze(), 
                            #     y.detach().cpu().squeeze()
                            # ]).permute(0,2,1))
                            # samples_mmse_y.append(torch.hstack([
                            #     X_posterior_mean.detach().cpu().squeeze(),
                            #     y.detach().cpu().squeeze()
                            # ]).permute(0,2,1))
                            # samples_last_y.append(torch.hstack([
                            #     last_sample.detach().cpu().squeeze(),
                            #     y.detach().cpu().squeeze()
                            # ]).permute(0,2,1))
                            
                            # save
                            psnr_temp = metrics.psnr_function(X_posterior_mean, x)
                            mse_temp = metrics.mse_function(X_posterior_mean, x)
                            ssim_temp = metrics.ssim_function(X_posterior_mean, x)
                            lpips_temp = metrics.lpips_function(X_posterior_mean, x)
                            matrics_pd = pd.DataFrame({
                                "name": [im_name],
                                "psnr": [psnr_temp],
                                "mse": [mse_temp],
                                "ssim": [ssim_temp],
                                "lpips": [lpips_temp],
                                "time": [collapsed_time]
                            })
                            save_dir_metric = os.path.join(save_metric_path, 'metrics_results.csv')
                            matrics_pd.to_csv(save_dir_metric, mode='a', header=not os.path.exists(save_dir_metric))
                            # del matrics_pd
                            s_lpips += lpips_temp
                            s_psnr += psnr_temp
                            s_mse += mse_temp
                            s_ssim += ssim_temp
                            count += 1
                            
                        # sliced wasserstein distance              
                        if args.evaluation.swd:    
                            # samples_ref = torch.stack(samples_ref_y)
                            # samples_mmse = torch.stack(samples_mmse_y)
                            # samples_last = torch.stack(samples_last_y)
                
                            if ii % args.evaluation.skipper == 0 and ii >= args.evaluation.skipper:
                                # save results as pandas dataframe
                                data_pd = pd.DataFrame({
                                    "Total images":[ii+1],
                                    # "swd mmse":[np.round(eval_swd(samples_ref, samples_mmse).item(),2)],
                                    # "swd last sample": [np.round(eval_swd(samples_ref, samples_last).item(),2)],
                                    "psnr mean":[s_psnr/count],
                                    "lpips mean":[s_lpips/count],
                                    "ssim mean":[s_ssim/count],
                                    "mse mean":[s_mse/count],
                                })
                                save_dir_swd = os.path.join(save_metric_path, 'swd_results.csv')
                                data_pd.to_csv(save_dir_swd, mode='a', header=not os.path.exists(save_dir_swd))
                                
                                delete([samples_last_y,samples_mmse_y,samples_ref_y])
                                # delete([samples_last,samples_last_y,samples_mmse_y,samples_ref_y,samples_mmse,samples_ref])
                        delete([X_posterior_mean,out])
                    

                        
                    # FiD
                    if args.evaluation.fid:
                        fid_mmse, fid_last = Fid_evatuation(root_dir_results, device)
        
                        print("after fid function !!!!")
                        fid_pd = pd.DataFrame({
                                                "fid mmse":[fid_mmse],
                                            "fid last":[fid_last]
                                            })
                        save_dir_fid = os.path.join(save_metric_path, 'fid_results.csv')
                        fid_pd.to_csv(save_dir_fid, mode='a', header=not os.path.exists(save_dir_fid))
                    
                    psnr_paths = np.mean(np.array(psnr_paths), axis=0)
                    psnr_paths_pd = pd.DataFrame({
                                                "psnr_paths":psnr_paths
                                            })
                    save_dir_psnr_paths = os.path.join(save_metric_path, 'psnr_paths.csv')
                    psnr_paths_pd.to_csv(save_dir_psnr_paths, mode='a', header=not os.path.exists(save_dir_psnr_paths))
                    
                    wandb.finish()    
if __name__ == "__main__":
    for solver in SOLVER_TYPES:
        for steps in NUM_TIMESTEPS:
            ddpm_params = [3] if solver == "ddpm" else [1] #[1, 1.5, 2, 2.5, 3]
            for ddpm_param in ddpm_params:
                main(steps, solver, ddpm_param)
