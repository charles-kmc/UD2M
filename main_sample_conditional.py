import os
import copy
import time
import numpy as np
import pandas as pd
import blobfile as bf
import matplotlib.pyplot as plt
import datetime
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from args_unfolded_model import args_unfolded
from models.load_model import load_frozen_model
from datasets.datasets import GetDatasets

import utils as utils
import unfolded_models as ums
import physics as phy
import runners as runners
from configs.args_parse import configs

from metrics.coverage.coverage_function import coverage_eval
from metrics.metrics import Metrics
from metrics.fid_batch.fid_evaluation import Fid_evatuation


def linear_indices(num_iter, lambda_min=0.3,lambda_max=0.6):
    seq = np.sqrt(np.linspace(0, num_iter**2, num_iter))
    seq = [int(s) for s in list(seq)]
    seq[-1] = seq[-1] - 1
    seq = np.abs(np.array(seq) - num_iter)
    seq[0] -=1
    seq[-1] -=1
    
    if num_iter>1:
        lambd = np.round(np.linspace(lambda_min, lambda_max, num_iter),2)
    else:
        lambd = np.array([0.32])
    lambd = lambd.reshape(1,-1)
    return lambd, seq
max_images = 0
max_iter = 1
ddpm_param = 1
run_name = "_sr_sampling"
# LAMBDA, INDICES = linear_indices(max_iter)
stochastic_model = False
stochastic_rate = 0.05
SOLVER_TYPES =  ["huen"]
LORA = True
MAX_UNFOLDED_ITER = [3]
NUM_TIMESTEPS = [100]#[50, 100, 200, 300, 500, 700, 800, 1000]
eta =1.0   # sr=0.8
T_AUG = 0
ZETA = [0.6]#[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]#[0.1] #op 0.4 or 0.6 sr = 0.9
date_eval = datetime.datetime.now().strftime("%d-%m-%Y")

def main(
    num_timesteps:int, 
    solver_type:str, 
    ddpm_param:float = 1.,
    date_eval:str = date_eval, 
    max_iter:int = 1, 
    eta:float=0.0
):
    with torch.no_grad():
        # args
        mode = "inference"
        args = args_unfolded()
        args.mode = mode
        config = configs(args.mode)
        args.task = config.task
        args.physic.operator_name = config.operator_name
        args.physic.sigma_model = config.sigma_model
        if args.mode == mode:
            ckpt_epoch = config.ckpt_epoch
            ckpt_date = config.ckpt_date
        if config.task == "inp":
            args.dpir.use_dpir = config.use_dpir
        args.lambda_ = config.lambda_
        args.lora = LORA
        args.evaluation.coverage = False
        args.ddpm_param = ddpm_param
        args.solver_type = solver_type
        args.eta = eta
        
        # lora dir and name
        args.lora_checkpoint_dir = bf.join(args.path_save, args.task, "Checkpoints", ckpt_date, f"model_noise_{args.physic.sigma_model}") 
        if args.task == "deblur" and args.physic.operator_name == "gaussian":
            args.lora_checkpoint_name = f"LoRA_model_{args.task}_{(ckpt_epoch):03d}.pt"            
        else:
            args.lora_checkpoint_name = f"LoRA_model_{args.task}_{args.physic.operator_name}_{(ckpt_epoch):03d}.pt"   
        
        if args.task=="sr":
            sr_f = 1#args.physic.sr.sf
            args.lora_checkpoint_name = f"LoRA_model_{args.task}_factor_{sr_f}_{(ckpt_epoch):03d}.pt"
            
        # parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        for idx in range(len(ZETA)):
            zeta = ZETA[idx]
            args.zeta = zeta
            for max_unfolded_iter in MAX_UNFOLDED_ITER:
                args.max_unfolded_iter = max_unfolded_iter
                args.date = ckpt_date
                args.epoch = ckpt_epoch
                args.eval_date = date_eval
                args.save_wandb_img = False                
                
                if args.dpir.use_dpir and args.task!="inp":
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
                    elif solver_type=="huen":
                        solver_params_title = f"huen"
                    else:
                        raise ValueError(f"Solver {solver_type} do not exists.")
                    wandb.init(
                            # set the wandb project where this run will be logged
                            project=f"Sampling Unfolded Conditional Diffusion Models {args.task}_{args.physic.operator_name}",
                            dir = dir_w,
                            config={
                                "model": "Deep Unfolding",
                                "Structure": "HQS",
                                "dataset": "FFHQ - 256",
                                "dir": dir_w,
                                "pertub":args.pertub,
                            },
                            name = f"{formatted_time}{run_name}_{solver_type}_num_steps_{num_timesteps}_unfolded_iter_{args.max_unfolded_iter}_{init}_max_iter_{max_iter}_task_{args.task}_rank_{args.rank}_{solver_params_title}_epoch_{args.epoch}_T_AUG_{T_AUG}",
                        )
                # data
                dir_test = "/users/cmk2000/sharedscratch/Datasets/testsets/imageffhq" #/users/cmk2000/sharedscratch/Datasets/testsets/ffhq or imageffhq
                datasets = GetDatasets(dir_test, args.im_size, dataset_name=args.dataset_name)
                testset = DataLoader(datasets, batch_size=1, shuffle=False, )
                
                # model
                frozen_model_name = 'diffusion_ffhq_10m' 
                frozen_model_dir = '/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo' 
                frozen_model_path = os.path.join(frozen_model_dir, f'{frozen_model_name}.pt')
                frozen_model, _ = load_frozen_model(frozen_model_name, frozen_model_path, device)
                
                # LoRA model 
                model = copy.deepcopy(frozen_model)

                target_lora = ["qkv", "proj_out"]
                if LORA:
                    utils.LoRa_model(
                        model, 
                        target_layer = target_lora, 
                        device=device, 
                        rank = args.rank
                    )             
                    runners.load_trainable_params(model, args)
                    
                # stochastic model setting
                target_lora_layer = ["qkv", "proj_out", "output_blocks.2.0.out_layers.3"]
                args.stochastic_model = stochastic_model
                if stochastic_model:
                    orig_params = {}
                    for name, param in model.named_parameters():
                        if name.endswith(tuple(target_lora_layer)):
                            orig_params[name] = param.data.clone()
                    args.orig_params = orig_params
                    args.target_lora_layer = target_lora_layer
                
                # Diffusion noise
                diffusion_scheduler = ums.DiffusionScheduler(device=device,noise_schedule_type=args.noise_schedule_type)
       
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
                        operator_name=args.physic.operator_name,
                    )
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
                        scale_image=args.physic.transform_y,
                        mode=args.mode,
                    )
                elif args.task=="sr":
                    physic = phy.SuperResolution(
                        im_size=args.im_size,
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
                    
                # HQS module
                denoising_timestep = ums.GetDenoisingTimestep(device)
                hqs_module = ums.HQS_models(
                    model,
                    physic, 
                    diffusion_scheduler, 
                    denoising_timestep,
                    args
                )
                    
                # sampler
                diffsampler = runners.Conditional_sampler(
                    hqs_module, 
                    physic, 
                    diffusion_scheduler, 
                    device, 
                    args,
                    max_unfolded_iter = max_unfolded_iter
                )

                # Metrics
                metrics = Metrics(device=device)
                
                # directories
                if args.lora:
                    use_lora = "lora"
                else:
                    use_lora = "no_lora"
                
                if solver_type == "ddim":
                    param_name_folder = os.path.join(f"zeta_{zeta}", f"eta_{eta}")
                elif solver_type == "ddpm":
                    param_name_folder = f"ddpm_param_{args.ddpm_param}"
                elif solver_type=="huen":
                    param_name_folder = "huen"
                else:
                    raise ValueError(f"Solver {solver_type} do not exists.")
                
                args.stochastic_rate = stochastic_rate
                
                root_dir_results = os.path.join(args.path_save, args.task, f"Results_{args.physic.operator_name}", args.eval_date, solver_type, use_lora, f"Max_iter_{max_iter}", f"unroll_iter_{args.max_unfolded_iter}", f"timesteps_{num_timesteps}", f"Epoch_{args.epoch}", f"T_AUG_{T_AUG}", param_name_folder, f"stochastic_{args.stochastic_rate}", f"lambda_{args.lambda_}")
                
                if args.evaluation.coverage or args.evaluation.metrics:
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
                s_psnr = 0
                s_mse = 0
                s_ssim = 0
                s_lpips = 0
                count = 0   
                results = {} 
                
                # Blur kernel 
                if args.task == "deblur" or args.task=="sr":    
                    blur = kernels.get_blur(seed=1234)
                    
                # loop over testset 
                for ii, im in enumerate(testset):
                    if ii>max_images:
                        continue
                        
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
                    elif args.task=="inp":
                        y = physic.y(im)
                    
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
                            x_true = x_true,
                            track = max_iter>1,
                            lambda_ = args.lambda_
                        )
                        # collapsed time
                        collapsed_time = time.time() - start_time           
                        # sample
                        if it ==0:
                            posterior = utils.welford(out["xstart_pred"])
                        if it>0:
                            # posterior
                            posterior.update(out["xstart_pred"])
                            # coverage 
                            if args.evaluation.coverage:
                                data[it-1] = out["xstart_pred"].detach().squeeze()
                            if args.use_wandb and ii%10==0 and args.evaluation.coverage:
                                psnr_val = metrics.psnr_function(out["xstart_pred"], x)
                                wandb.log({f"sample psnr {im_name}": psnr_val})
                    # posterior mean - mmse
                    if max_iter>1:
                        X_posterior_mean = posterior.get_mean()
                        X_posterior_var = posterior.get_var()
                    else:
                        X_posterior_mean = out["xstart_pred"]
                        X_posterior_var = out["xstart_pred"]
                    utils.delete([posterior])
                    
                    # psnrs 
                    try:
                        fig = plt.figure()
                        plt.plot(range(len(out["psnrs"])), out["psnrs"], label = f"psnr {im_name}")
                        plt.xlabel("T")
                        plt.ylabel("psnr")
                        plt.savefig(os.path.join(metric_path, f"psnr_progress_{im_name}.png"))
                    except:
                        pass
                        
                    if args.use_wandb:
                        psnr_val = metrics.psnr_function(out["xstart_pred"], x)
                        wandb.log({f"psnrs": psnr_val}) 
                        
                    if args.use_wandb and ii%20 == 0:
                        im_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(x), 
                            caption=f"true image", 
                            file_type="png"
                        )
                        x0_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(out["xstart_pred"]), 
                            caption=f"sample psnr/mse ({metrics.psnr_function(out["xstart_pred"], x):.2f},{metrics.lpips_function(out["xstart_pred"], x):.2f}))", 
                            file_type="png"
                        )
                        xmmse_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(X_posterior_mean), 
                            caption=f"x_mmse psnr/lpips ({metrics.psnr_function(X_posterior_mean, x):.2f},{metrics.lpips_function(X_posterior_mean, x):.2f})", 
                            file_type="png"
                        )
                        xvar_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(X_posterior_var.sqrt() / X_posterior_var.sqrt().max()), 
                            caption=f"standard deviation", 
                            file_type="png"
                        )
                        z0_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(out["aux"]), 
                            caption=f"z_est psnr/mse ({metrics.psnr_function(out["aux"], x):.2f},{metrics.lpips_function(out["aux"], x):.2f})", 
                            file_type="png"
                        )
                        y_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(utils.inverse_image_transform(y)), 
                            caption=f"observation", 
                            file_type="png"
                        )
                        xy = [im_wdb, y_wdb, x0_wdb, z0_wdb, xmmse_wdb, xvar_wdb]
                        wandb.log({"blurred and predict":xy})
                        # progression
                        image_wdb = wandb.Image(out["progress_img"], caption=f"progress sequence for im {im_name}", file_type="png")
                        wandb.log({"pregressive":image_wdb})
                        utils.delete([image_wdb, xy, xmmse_wdb, z0_wdb, y_wdb, x0_wdb, im_wdb, xvar_wdb])
                    
                    # coverage
                    if args.evaluation.coverage:
                        op_coverage = utils.DotDict()
                        op_coverage.inside_interval = []; op_coverage.num_dist_greater = []
                        op_coverage.im_num = [ii]; op_coverage.truexdist_list = []; op_coverage.quantilelist_list = []
                        op_coverage.save_dir = save_metric_path
                        op_coverage.data = data
                        op_coverage.post_mean = X_posterior_mean.detach().cpu()
                        alpha_levels = np.round(np.arange(0.9, 1, 0.02),2).tolist()
                        alpha_levels.append(0.99)
                        alpha_levels.append(0.999)
                        op_coverage.prob = np.array([alpha_levels])
                        out_coverage = coverage_eval(x.detach().cpu(), op_coverage)
                    
                    last_sample = data[-1].to(device) if args.evaluation.coverage else out["xstart_pred"]
                    utils.delete([data, op_coverage]) if args.evaluation.coverage else None
                    
                    if args.evaluation.coverage:
                        sampledist = out_coverage["sampledist"]
                        truexdist = out_coverage["truexdist"]
                        results[f"{ii}_sample_dist"] = sampledist
                        results[f"{ii}_truex_dist"] = truexdist
                        plt.figure(figsize=(8, 6))
                        plt.hist(sampledist, bins=30, density=True, alpha=0.7, color='blue')
                        plt.xlabel('Value')
                        plt.ylabel('Density')
                        plt.title('Histogram of the sample distance from the MMSE')
                        plt.grid(True, alpha=0.3)
                        # Add a line showing the theoretical uniform distribution
                        plt.axvline(x=truexdist, color='r', linestyle='--', label='True distance')
                        plt.legend()
                        plt.savefig(os.path.join(metric_path, f"sampledist_{im_name}.png"))

                    # sliced wasserstein distance
                    if args.evaluation.save_images:   
                        # save images for fid and cmmd evaluation
                        utils.save_images(ref_path, x, im_name)
                        utils.save_images(mmse_path, X_posterior_mean, im_name)
                        std = torch.sqrt(X_posterior_var)
                        err = torch.abs(X_posterior_mean-x)
                        norm = max(std.max(), err.max())
                        utils.save_images(var_path, std/norm, im_name)
                        utils.save_images(var_path, err /norm, im_name+"_residual")
                        utils.save_images(y_path, utils.inverse_image_transform(y), im_name)
                        utils.save_images(last_path, last_sample, im_name)
                        
                        results[f"{ii}_var"] = X_posterior_var.cpu().squeeze().numpy()
                        results[f"{ii}_mmse"] = X_posterior_mean.cpu().squeeze().numpy()
                        results[f"{ii}_ref"] = x.cpu().squeeze().numpy()
                        results[f"{ii}_last"] = last_sample.cpu().squeeze().numpy()
                        sio.savemat(os.path.join(save_metric_path, "variance_results.mat"), results)
                        
                        # eval - mmse
                        psnr_temp = metrics.psnr_function(X_posterior_mean, x)
                        mse_temp = metrics.mse_function(X_posterior_mean, x)
                        ssim_temp = metrics.ssim_function(X_posterior_mean, x)
                        lpips_temp = metrics.lpips_function(X_posterior_mean, x)
                        
                        psnr_temp_last = metrics.psnr_function(last_sample, x)
                        mse_temp_last = metrics.mse_function(last_sample, x)
                        ssim_temp_last = metrics.ssim_function(last_sample, x)
                        lpips_temp_last = metrics.lpips_function(last_sample, x)
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
                        # del matrics_pd
                        s_lpips += lpips_temp
                        s_psnr += psnr_temp
                        s_mse += mse_temp
                        s_ssim += ssim_temp
                        count += 1
                    utils.delete([X_posterior_mean,out])
                
                wandb.finish()  
    return args 


if __name__ == "__main__":

    for solver in SOLVER_TYPES:
        for steps in NUM_TIMESTEPS:
            for eta in [0.8]:# [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
                args = main(steps, solver, max_iter=max_iter,eta= eta)
    args.step = steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_timestep = args.step
    param_name_folder = os.path.join(f"zeta_{args.zeta}", f"eta_{args.eta}")
    
    dic_results = {}
    vec_fid_mmse = []
    vec_psnr_mmse = []
    vec_ssim_mmse = []
    vec_lpips_mmse = []
    vec_fid_last = []
    if max_images>50:
        for num_timestep in NUM_TIMESTEPS:
            for lambda_ in [args.lambda_]:
                root_dir = os.path.join(args.path_save, args.task, f"Results_{args.physic.operator_name}", args.eval_date, "ddim", "lora", f"Max_iter_{max_iter}", f"unroll_iter_{args.max_unfolded_iter}", f"timesteps_{num_timestep}", f"Epoch_{args.epoch}", f"T_AUG_{T_AUG}", param_name_folder, f"stochastic_{args.stochastic_rate}", f"lambda_{lambda_}")
                save_metric_path = os.path.join(args.path_save, args.task, f"Results_{args.physic.operator_name}", args.eval_date, "ddim", "lora", f"Max_iter_{max_iter}", f"unroll_iter_{args.max_unfolded_iter}", f"timesteps_{num_timestep}", f"Epoch_{args.epoch}", f"T_AUG_{T_AUG}", param_name_folder, f"stochastic_{args.stochastic_rate}", "metrics_evaluated")
                root_dir_csv = os.path.join(root_dir, "metrics_results")
                # FID
                out= Fid_evatuation(root_dir, device)
                vec_fid_mmse.append(out["fid_mmse"])
                vec_fid_last.append(out["fid_last"])
                dic_results[f"fid mmse_timesteps{lambda_}"] = [out["fid_mmse"]]
                dic_results[f"fid last_timesteps{lambda_}"] = [out["fid_last"]]
                
                # other metrics: PSNR, SSIM, LPIPs
                data = pd.read_csv(os.path.join(root_dir_csv, "metrics_results.csv"))
                mean_psnr = data["psnr"].mean()
                mean_lpips = data["lpips"].mean()
                mean_ssim = data["ssim"].mean()
                dic_results[f"psnr mmse_timesteps{lambda_}"] = [mean_psnr]
                dic_results[f"lpips mmse_timesteps{lambda_}"] = [mean_lpips]
                dic_results[f"ssim mmse_timesteps{lambda_}"] = [mean_ssim]
                vec_lpips_mmse.append(mean_lpips)
                vec_psnr_mmse.append(mean_psnr)
                vec_ssim_mmse.append(mean_ssim)
            
            # save data in a csv file 
            lab ="T"
            os.makedirs(save_metric_path, exist_ok=True)
            fid_pd = pd.DataFrame(dic_results)
            save_dir_metrics = os.path.join(save_metric_path, f'fid_psnr_lpips_ssim_results.csv') #_eta_{eta}_zeta_{zeta}
            fid_pd.to_csv(save_dir_metrics, mode='a', header=not os.path.exists(save_dir_metrics))
            
            # plot the results
            try:
                xaxis = NUM_TIMESTEPS
                if len(xaxis) >1:
                    plot = plt.figure()
                    plt.plot(np.array(xaxis), np.array(vec_fid_mmse), "b.", label = "FID - mmse")
                    plt.plot(np.array(xaxis), np.array(vec_fid_last), "b.", label = "FID - last")
                    plt.xlabel(f"{lab}")
                    plt.ylabel("FID")
                    plt.title(f"{lab} versus FID")
                    plt.legend()
                    plt.savefig(os.path.join(save_metric_path, f"fid_results.png"))
                    plt.close("all")
                    
                    # plot the results
                    plot = plt.figure()
                    plt.plot(np.array(xaxis), np.array(vec_lpips_mmse), "b.", label = "lpips")
                    plt.legend()
                    plt.xlabel(f"{lab}")
                    plt.ylabel("LPIPs")
                    plt.title(f"{lab} versus lpips")
                    plt.savefig(os.path.join(save_metric_path, f"lpips_results.png")) # _eta_{eta}_zeta{zeta}
                    plt.close("all")
                    
                    # plot the results
                    plot = plt.figure()
                    plt.plot(np.array(xaxis), np.array(vec_psnr_mmse), "b.", label = "psnr")
                    plt.legend()
                    plt.xlabel(f"{lab}")
                    plt.ylabel("PSNR")
                    plt.title("zeta versus psnr")
                    plt.savefig(os.path.join(save_metric_path, f"psnr_results.png")) # _eta_{eta}_zeta{zeta}
                    plt.close("all")
                    
                    # plot the results
                    plot = plt.figure()
                    plt.plot(np.array(xaxis), np.array(vec_ssim_mmse), "b.", label = "ssim")
                    plt.legend()
                    plt.xlabel(f"{lab}")
                    plt.ylabel("SSIM")
                    plt.title(f"{lab} versus ssim")
                    plt.savefig(os.path.join(save_metric_path, f"ssim_results.png")) #_eta_{eta}_zeta{zeta}
                    plt.close("all")
            except:
                pass
            
    # evaluation 
    if False:   
        kargs = DotDict(
            {
                "NUM_TIMESTEPS":NUM_TIMESTEPS,  
                "ZETA":ZETA,
                "ddpm_params":ddpm_params,
                "eta":eta,
                "T_AUG":T_AUG,
                "date":date_eval,
                "epoch":ckpt_epoch,
                "max_iter":max_iter,
                "task":"debur",
                "max_unfolded_iter":3
            }
        )
    
        xaxis = NUM_TIMESTEPS
        lab = "timesteps"
        if solver == "ddim" and len(NUM_TIMESTEPS)>1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            date = kargs.date_eval
            epoch = kargs.epoch
            max_iter = kargs.max_iter
            task = kargs.task 
            max_unfolded_iter = kargs.max_unfolded_iter 
            
            dic_results = {}
            vec_fid_mmse = []
            vec_psnr_mmse = []
            vec_ssim_mmse = []
            vec_lpips_mmse = []
            
            for timesteps in NUM_TIMESTEPS:
                zeta = ZETA[0]
                root_dir = f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/{task}/Results/{date}/{solver}/lora/Max_iter_{max_iter}/unroll_iter_{max_unfolded_iter}/timesteps_{timesteps}/Epoch_{epoch}/T_AUG_{T_AUG}/zeta_{zeta}/eta_{eta}"
                save_metric_path = os.path.join(f"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp/{task}/Results/{date}/{solver}/lora/Max_iter_{max_iter}/unroll_iter_{max_unfolded_iter}", f"metrics_results_calib_timesteps_{lab}")
                
                # FID
                fid_mmse, _ = Fid_evatuation(root_dir, device)
                vec_fid_mmse.append(fid_mmse)
                dic_results[f"fid mmse_timesteps{zeta}"] = [fid_mmse]
                
                # other metrics: PSNR, SSIM, LPIPs
                root_dir_csv = os.path.join(root_dir, "metrics_results")
                data = pd.read_csv(os.path.join(root_dir_csv, "swd_results.csv"))
                mean_psnr = data["psnr mean"].mean()
                mean_lpips = data["lpips mean"].mean()
                mean_ssim = data["ssim mean"].mean()
                dic_results[f"psnr mmse_timesteps{zeta}"] = [mean_psnr]
                dic_results[f"lpips mmse_timesteps{zeta}"] = [mean_lpips]
                dic_results[f"ssim mmse_timesteps{zeta}"] = [mean_ssim]
                vec_lpips_mmse.append(mean_lpips)
                vec_psnr_mmse.append(mean_psnr)
                vec_ssim_mmse.append(mean_ssim)
            
            
            # save data in a csv file 
            os.makedirs(save_metric_path, exist_ok=True)
            fid_pd = pd.DataFrame(dic_results)
            save_dir_metrics = os.path.join(save_metric_path, f'fid_psnr_lpips_ssim_results_T_AUG_{T_AUG}_ddim_{lab}.csv') #_eta_{eta}_zeta_{zeta}
            fid_pd.to_csv(save_dir_metrics, mode='a', header=not os.path.exists(save_dir_metrics))
            
            # plot the results
            plot = plt.figure()
            plt.plot(np.array(xaxis), np.array(vec_fid_mmse), "b.", label = "FID")
            plt.xlabel(f"{lab}")
            plt.ylabel("FID")
            plt.title(f"{lab} versus FID")
            plt.legend()
            plt.savefig(os.path.join(save_metric_path, f"fid_results_T_AUG_{T_AUG}_eta_{eta}_zeta_{zeta}.png"))
            plt.close("all")
            
            # plot the results
            plot = plt.figure()
            plt.plot(np.array(xaxis), np.array(vec_lpips_mmse), "b.", label = "lpips")
            plt.legend()
            plt.xlabel(f"{lab}")
            plt.ylabel("LPIPs")
            plt.title(f"{lab} versus lpips")
            plt.savefig(os.path.join(save_metric_path, f"lpips_results_T_AUG_{T_AUG}.png")) # _eta_{eta}_zeta{zeta}
            plt.close("all")
            
            # plot the results
            plot = plt.figure()
            plt.plot(np.array(xaxis), np.array(vec_psnr_mmse), "b.", label = "psnr")
            plt.legend()
            plt.xlabel(f"{lab}")
            plt.ylabel("PSNR")
            plt.title("zeta versus psnr")
            plt.savefig(os.path.join(save_metric_path, f"psnr_results_T_AUG_{T_AUG}.png")) # _eta_{eta}_zeta{zeta}
            plt.close("all")
            
            # plot the results
            plot = plt.figure()
            plt.plot(np.array(xaxis), np.array(vec_ssim_mmse), "b.", label = "ssim")
            plt.legend()
            plt.xlabel(f"{lab}")
            plt.ylabel("SSIM")
            plt.title(f"{lab} versus ssim")
            plt.savefig(os.path.join(save_metric_path, f"ssim_results_T_AUG_{T_AUG}.png")) #_eta_{eta}_zeta{zeta}
            plt.close("all")
    

