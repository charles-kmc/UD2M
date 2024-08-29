import torch 
from unfolded_models.DiffUNet import DiffUNet
from unfolded_utils.load_problem import get_dataset
import deepinv as dinv
from unfolded_models.unrolledPhysics import DiffJointPhysics, DiffusionNoise, ckpt_denoiser
from unfolded_models.HQSIteration import init_hqs_diff, HQSDiff
from unfolded_utils.trainer import DiffusionTrainer
from unfolded_utils.unfolded_utils import unfolded_builderLP
from unfolded_utils.dinv_metrics import LPIPS_seed, SSIM_seed

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

## Training Params 
BATCH_SIZE      =       8
LR_optim        =       1e-3
LR_UNet         =       1e-4
EPOCHS          =       2


## Get datasets
train_set, test_set, img_shape = get_dataset(
    datasetname="FFHQ",
    img_width=256
)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size = BATCH_SIZE,
    num_workers = 4,
    shuffle = True,
    drop_last = True
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size = BATCH_SIZE,
    num_workers = 4,
    shuffle = False
)

LORA = True
denoiser = DiffUNet(
                device=device,
                use_lora=LORA,  ## Switch on/off LORA
                model_path='../sharedscratch/modelzoo/diffusion_ffhq_10m.pt'
            )
denoiser = ckpt_denoiser(denoiser)
prior = dinv.optim.PnP(denoiser = denoiser)
data_fidelity = dinv.optim.data_fidelity.L2()

## Get Measurement and diffusion noise operators
noise_level = 0.05
kernel = dinv.physics.blur.gaussian_blur(sigma=(3,3))
y_noise = dinv.physics.GaussianNoise(sigma=noise_level)
y_operator = dinv.physics.BlurFFT(
    img_size=img_shape,
    filter=kernel,
    device=device,
    noise_model=y_noise
) # Forward calls take x0 and return y
diffusion_operator = DiffusionNoise(
    T=1000,
    beta_start=0.1/1000.,
    beta_end=20./1000.,
    device=device
) # Forward calls take x0 and return xt
Joint_operator = DiffJointPhysics(
    y_physics=y_operator, 
    diff_physics=diffusion_operator
)  # Forward calls take x0 and return (y, xt)


## Optimization Params
iterator = HQSDiff()
num_optim_iterations = 3
optim_params = {
     "stepsize":[noise_level**2]*num_optim_iterations  # Data fidelity proximal scaling parameter for each step
}

## Unfolded Optimization scheme with UNet prior
unfolded_model = unfolded_builderLP(
     iteration=HQSDiff(),
     params_algo=optim_params,
     custom_init=init_hqs_diff,
     trainable_params=optim_params.copy(),
     log_trainable_params=optim_params.copy(),  # Learn log of optim params
     data_fidelity=data_fidelity,
     prior=prior,
     device=device,
     max_iter=num_optim_iterations
)

def train():
    wandb_config = {
    "project":"Conditional_Diffusion_for_IV",
    "name":"Unrolled_HQS_UNet",
    "config":{
            "LORA":LORA
        }
    }
    optimizer_params = []  # Parameters for the optimization algorithm
    UNet_params = []  # Parameters in the UNet
    print("Trainable Parameters")
    for k, p in unfolded_model.named_parameters():
        if p.requires_grad:
            if "prior." in k:
                print(f"prior: {k}")
                UNet_params.append(p)
            else:
                print(f"Optim: {k}")
                optimizer_params.append(p)

    losses = [dinv.loss.SupLoss()]
    metrics = [dinv.loss.PSNR(), LPIPS_seed(device=device), SSIM_seed(device=device)]
    optimizer = torch.optim.Adam(
         [
            {"params":optimizer_params, "lr":LR_optim, "tag":"Optimparams"},
            {"params":UNet_params, "lr":LR_UNet, "tag":"UNetparams"}
         ],
         weight_decay=0
    )
    trainer = DiffusionTrainer(
        unfolded_model,
        save_every_n_iter=50,
        training_metrics=[dinv.loss.PSNR()],
        physics = Joint_operator,
        train_dataloader=train_loader,
        eval_dataloader = test_loader,
        epochs = EPOCHS,
        losses=losses,
        optimizer=optimizer,
        device=device,
        save_path = './unfolded_models/saved/HQS',
        verbose = 1,
        show_progress_bar = True,
        plot_images=True,
        plot_metrics=True,
        online_measurements=True,
        metrics=metrics,# Removed SSIM & LPIPS for now - pending rng bug fix, dinv.loss.SSIM(device=device), dinv.loss.LPIPS(device=device)],
        wandb_vis=True,
        wandb_setup=wandb_config,
    )


    model = trainer.train(epochs=EPOCHS, no_expand_levels=0)
    return model


if __name__ == "__main__":
    train()