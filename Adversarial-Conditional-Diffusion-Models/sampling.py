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
from metrics.coverage.coverage_function import coverage_eval
from metrics.metrics import Metrics
from datasets.get_datasets import get_dataset, get_dataloader


max_images = 5

max_iter = 1
ddpm_param = 1

stochastic_model = False
stochastic_rate = 0.05
SOLVER_TYPES =  ["ddim"] # huen, ddpm, ddim
LORA = True

etas =[0.8]  
T_AUG = 0
ZETA = [0.9]
eval_date = datetime.datetime.now().strftime("%d-%m-%Y")

def main(
    solver_type:str, 
    ddpm_param:float = 1.,
    eta:float=0.0,
    **kwargs,
):
    with torch.no_grad():
        # args
        mode = "inference"
        
        script_dir = os.path.dirname(__file__)
        # args
        config = configs(mode=mode)
        print(os.path.join(script_dir,"configs", config.config_file))
        
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
        #args.dpir.use_dpir = False
        run_name = f"_{args.task}_sampling"
        args.lora_checkpoint_dir = config.ckpt_dir
        args.lora_checkpoint_name = config.ckpt_name
        max_images = config.max_images
        args.physic.kernel_size = config.kernel_size
        
        # parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        args.device = device
        
        for idx in range(len(ZETA)):
            zeta = ZETA[idx]
            args.zeta = zeta
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
                    dir_w="/users/cmk2000/sharedscratch/Results/wandb"
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
                            name = f"{formatted_time}{run_name}_{solver_type}_num_steps_{num_timesteps}_unfolded_iter_{config.max_unfolded_iter}_task_{args.task}_rank_{args.model.rank}_{solver_params_title}",
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
                
                print(f"Dataset {args.data.dataset_name} loaded with {len(loader)} images.\n\n\n\n")
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
                        (3,256, 256),
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
                hqs_module.load_state_dict(trainable_state_dict["HQS_state_dict"], strict=False)
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
                    state_dict_RAM=trainable_state_dict["RAM_state_dict"]
                )

                # Metrics
                metrics = Metrics(device=device)
                
                # directories
                if args.model.use_lora:
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
                
                if "generalised" in lora_checkpoint_dir:
                    root_name = f"_RAM_unfolded_iter_{config.max_unfolded_iter}_generalised_rank_{args.model.rank}_Paper"
                else:
                    root_name = f"_RAM_unfolded_iter_{config.max_unfolded_iter}_rank_{args.model.rank}_paper_new"
                    
                root_save= os.path.join(args.save_dir+root_name, 
                                args.data.dataset_name, 
                                f"{args.task}_{use_lora}", 
                                f"operator_{args.physic.operator_name}_{args.physic.kernel_size}", 
                                f"var_{args.physic.sigma_model}", 
                                eval_date, 
                                solver_type
                )
                
                if args.task=="inp":                
                    root_dir_results = os.path.join(
                                root_save, 
                                f"scale_{args.physic.inp.mask_rate}", 
                                f"timesteps_{num_timesteps}", 
                                param_name_folder, 
                                f"lambda_{args.lambda_}"
                    )
                elif args.task=="sr":
                    root_dir_results = os.path.join(
                                root_save, 
                                f"factor_{args.physic.sr.sf}", 
                                f"timesteps_{num_timesteps}", 
                                param_name_folder, 
                                f"lambda_{args.lambda_}"
                    )
                elif args.task=="deblur":
                    root_dir_results = os.path.join(
                                root_save, 
                                f"timesteps_{num_timesteps}", 
                                param_name_folder, 
                                f"lambda_{args.lambda_}"
                    )
                elif args.task=="jpeg":
                    root_dir_results = os.path.join(
                                root_save, 
                                f"timesteps_{num_timesteps}", 
                                f"qf_{args.physic.jpeg.qf}", 
                                f"lambda_{args.lambda_}"
                    )
                else:
                    raise ValueError(f"Task {args.task} do not exists.")
                
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
                # results = {} 
                
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
                    # if ii*im.shape[0]>max_images:
                    # if ii*im.shape[0] not in [47, 113, 297]:continue
                        # break
                    if ii not in [138,213,270,271,498,560,671,714,725,755,884,1356]:continue
                        
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
                        
                    if args.use_wandb and ii%2 == 0:
                        im_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(x[0].unsqueeze(0)), 
                            caption=f"true image", 
                            file_type="png"
                        )
                        x0_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(out["xstart_pred"][0].unsqueeze(0)), 
                            caption=f"sample psnr/mse ({metrics.psnr_function(out['xstart_pred'][0].unsqueeze(0), x[0].unsqueeze(0)):.2f},{metrics.lpips_function(out['xstart_pred'][0].unsqueeze(0), x[0].unsqueeze(0)):.2f}))", 
                            file_type="png"
                        )
                        xmmse_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(X_posterior_mean[0].unsqueeze(0)), 
                            caption=f"x_mmse psnr/lpips ({metrics.psnr_function(X_posterior_mean[0].unsqueeze(0), x[0].unsqueeze(0)):.2f},{metrics.lpips_function(X_posterior_mean[0].unsqueeze(0), x[0].unsqueeze(0)):.2f})", 
                            file_type="png"
                        )
                        xvar_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(X_posterior_var[0].unsqueeze(0).sqrt() / X_posterior_var[0].unsqueeze(0).sqrt().max()), 
                            caption=f"standard deviation", 
                            file_type="png"
                        )
                        z0_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(out["aux"][0].unsqueeze(0)), 
                            caption=f"z_est psnr/mse ({metrics.psnr_function(out['aux'][0].unsqueeze(0), x[0].unsqueeze(0)):.2f},{metrics.lpips_function(out['aux'][0].unsqueeze(0), x[0].unsqueeze(0)):.2f})", 
                            file_type="png"
                        )
                        y_wdb = wandb.Image(
                            utils.get_rgb_from_tensor(utils.inverse_image_transform(y[0].unsqueeze(0))), 
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
                        # results[f"{ii}_sample_dist"] = sampledist
                        # results[f"{ii}_truex_dist"] = truexdist
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
                        for iii, (x_t, xp_t, y_t, v_t) in enumerate(zip(x, X_posterior_mean, y, X_posterior_var)):
                            im_name = f"{(ii*x.shape[0] + iii):05d}"
                            # save images for fid and cmmd evaluation
                            utils.save_images(ref_path, x_t, im_name)
                            utils.save_images(mmse_path, xp_t, im_name)
                            # std = torch.sqrt(X_posterior_var)
                            std = v_t
                            err = torch.abs(xp_t-x_t)
                            norm = max(std.max(), err.max())
                            # utils.save_images(var_path, std/norm, im_name)
                            # utils.save_images(var_path, err /norm, im_name+"_residual")
                            utils.save_images(y_path, utils.inverse_image_transform(y_t), im_name)
                            # utils.save_images(last_path, last_sample, im_name)
                            
                            # results[f"{ii*im.shape[0] + iii}_var"] = v_t.cpu().squeeze().numpy()
                            # results[f"{ii*im.shape[0] + iii}_mmse"] = xp_t.cpu().squeeze().numpy()
                            # results[f"{ii*im.shape[0] + iii}_ref"] = x.cpu().squeeze().numpy()
                            # results[f"{ii*im.shape[0] + iii}_last"] = last_sample.cpu().squeeze().numpy()
                            # sio.savemat(os.path.join(save_metric_path, "variance_results.mat"), results)
                            
                            # eval - mmse
                            psnr_temp = metrics.psnr_function(xp_t, x_t)
                            mse_temp = metrics.mse_function(xp_t, x_t)
                            ssim_temp = metrics.ssim_function(xp_t, x_t)
                            lpips_temp = metrics.lpips_function(xp_t, x_t)
                            
                            psnr_temp_last = metrics.psnr_function(last_sample[iii], x_t)
                            mse_temp_last = metrics.mse_function(last_sample[iii], x_t)
                            ssim_temp_last = metrics.ssim_function(last_sample[iii], x_t)
                            lpips_temp_last = metrics.lpips_function(last_sample[iii], x_t)
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
            for eta in etas:
                kwargs = {
                    "solver_type": solver,
                    "eta": eta,
                }
             
                args = main(**kwargs)
