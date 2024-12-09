import os
import datetime
import torch
from torch.utils.data import DataLoader

from args_unfolded_model import args_unfolded
from models.load_model import load_frozen_model
from datasets.datasets import GetDatasets
import utils as utils
import runners as runners
import unfolded_models as unfolded_models

from configs.args_parse import configs

max_iter = 1
SOLVER_TYPES =  ["ddim"]
NUM_TIMESTEPS = [50, 100, 300, 400, 600, 700, 1000]#[20, 50, 100, 150, 200, 250, 350, 500, 600, 700, 800, 900, 1000]
eta = 0.0 
ZETA = [0.2] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
date_eval = datetime.datetime.now().strftime("%d-%m-%Y")

def main(num_timesteps, solver_type, ddpm_param = 1):
    with torch.no_grad():
        # args
        args = args_unfolded()
        args.task = configs().task
        
        args.ddpm_param = ddpm_param
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        args.solver_type = solver_type
        args.eta = eta
        
        for idx in range(len(ZETA)):
            zeta = ZETA[idx]
            args.zeta = zeta
            if args.use_wandb:
                import wandb
                formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dir_w = "/users/cmk2000/sharedscratch/Results/wandb"
                os.makedirs(dir_w, exist_ok=True)
                wandb.init(
                        # set the wandb project where this run will be logged
                        project=f"Sampling Diffusion Models linear scheduler {solver_type}",
                        dir = dir_w,
                        config={
                            "model": "Diffusion",
                            "dataset": "FFHQ - 256",
                            "dir": dir_w,
                        },
                        name = f"{formatted_time}_{solver_type}_num_steps_{num_timesteps}",
                    )
            # Test dataset
            dir_test = "/users/cmk2000/sharedscratch/Datasets/testsets/imageffhq"
            datasets = GetDatasets(dir_test, args.im_size, dataset_name=args.dataset_name)
            testset = DataLoader(datasets, batch_size=1, shuffle=False)
            
            # model
            pretrained_model_path = os.path.join(args.model.pretrained_model_path, f'{args.model.model_name}.pt')
            model, _ = load_frozen_model(args.model.model_name, pretrained_model_path, device)
                            
            #  diffusion scheduler and sampler module
            diffusion = unfolded_models.DiffusionScheduler(device=device)
            diffusion_sampler = runners.Unconditional_sampler(diffusion, model,device, args)

            # directories
            if solver_type == "ddim":
                param_name_folder = os.path.join(f"zeta_{zeta}", f"eta_{eta}")
            elif solver_type == "ddpm":
                param_name_folder = f"ddpm_param_{args.ddpm_param}"
            else:
                raise ValueError(f"Solver {solver_type} do not exists.")
            
            root_dir_results = os.path.join(args.path_save, args.task, "Results", date_eval, solver_type, "no_lora", "Simple sampling", f"timesteps_{num_timesteps}", param_name_folder)
            ref_path = os.path.join(root_dir_results, "ref")
            sample_path = os.path.join(root_dir_results, "last")
            for dir_ in [ref_path, sample_path]:
                os.makedirs(dir_, exist_ok=True)
            
            # loop over testset 
            for ii, im in enumerate(testset):
                if ii > 100:
                    continue
                im = im.to(device)
                x = utils.inverse_image_transform(im)
                im_name = f"{(ii):05d}"
                
                # reverse diffusion model
                out = diffusion_sampler.sampler(x.shape, num_timesteps = num_timesteps, eta=eta, zeta=zeta)

                if args.use_wandb and ii%20 == 0:
                    im_wdb = wandb.Image(utils.get_rgb_from_tensor(x), caption=f"true image", file_type="png")
                    x0_wdb = wandb.Image(utils.get_rgb_from_tensor(out["xstart_pred"]), caption=f"sample", file_type="png")
                    image_wdb = wandb.Image(out["progress_img"], caption=f"progress sequence for im {im_name}", file_type="png")
                    xy = [im_wdb, x0_wdb]
                    wandb.log({"blurred and predict":xy, "pregressive":image_wdb})
                    utils.delete([image_wdb, xy, x0_wdb, im_wdb])
                
                # save images
                utils.save_images(ref_path, x, im_name)
                utils.save_images(sample_path, out["xstart_pred"], im_name)
                utils.delete([out])
            
            wandb.finish()    

if __name__ == "__main__":
    for solver in SOLVER_TYPES:
        for steps in NUM_TIMESTEPS:
            ddpm_params = [3] if solver == "ddpm" else [1] 
            for ddpm_param in ddpm_params:
                main(steps, solver, ddpm_param)
        
        
        