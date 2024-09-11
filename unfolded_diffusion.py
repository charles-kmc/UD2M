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
EPOCHS          =       10


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

LORA = True ## Switch on/off LORA
denoiser = DiffUNet(
                device=device,
                use_lora=LORA,  ## Switch on/off LORA
                model_path='../sharedscratch/modelzoo/diffusion_ffhq_10m.pt'
            )
# denoiser = ckpt_denoiser(denoiser)  ## Checkpointing currently does not work with LORA
data_fidelity = dinv.optim.data_fidelity.L2()

## Get Measurement and diffusion noise operators
noise_level = 0.05

## Load Forward Operator on y
# Currently using Gaussian deblurring,
# however the code should run for any instance
# of deepinv.physics.DecomposablePhysics for which one can easily compute 
# the prox operator of the data fidelity
kernel = dinv.physics.blur.gaussian_blur(sigma=(3,3))  ## Use gaussian blur kernel
y_noise = dinv.physics.GaussianNoise(sigma=noise_level)  ## Noise model for y
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
num_optim_iterations = 1
optim_params = {
     "stepsize":[noise_level**2]*num_optim_iterations,  # Data fidelity proximal scaling parameter for each step
     "eta":[1.]
}


## Unfolded Optimization scheme with UNet prior
unfolded_model = unfolded_builderLP(
     iteration=HQSDiff(),
     params_algo=optim_params,
     custom_init=init_hqs_diff,  # Function to initialize HQS given (y, xt and the forward operator)
     trainable_params=optim_params.copy(),
     log_trainable_params=optim_params.copy(),  # Learn log of optim params
     data_fidelity=data_fidelity,
     prior=denoiser,
     device=device,
     max_iter=num_optim_iterations
)

def train():
    wandb_config = {
    "project":"Conditional_Diffusion_for_IVP",
    "name":"Unrolled_HQS_UNet",
    "config":{
            "LORA":LORA
        }
    }
    optimizer_params = []  # Parameters for the optimization algorithm
    UNet_params = []  # Parameters in the UNet
    time_embed_params = []
    print("Trainable Parameters")
    for k, p in unfolded_model.named_parameters():
        if p.requires_grad:
            if "prior." in k:
                print(f"prior: {k}")
                if "time_embed" in k:
                    time_embed_params.append(p)
                else:
                    UNet_params.append(p)

            else:
                print(f"Optim: {k}")
                optimizer_params.append(p)

    losses = [dinv.loss.SupLoss()]
    metrics = [dinv.loss.PSNR(), LPIPS_seed(device=device), SSIM_seed(device=device)]
    unrol_param_group = {"params":optimizer_params, "lr":LR_optim, "tag":"Optimparams"}
    UNET_param_group = {"params":UNet_params, "lr":LR_UNet, "tag":"UNetparams"}
    embedding_param_group = {"params":time_embed_params, "lr":LR_UNet, "tag":"Embeddingparams"}
    optimizer = torch.optim.Adam(
         [
            embedding_param_group
         ],
         weight_decay=0
    )  # Initialize optimizer with only parameters to be pretrained

    trainer = DiffusionTrainer(
        unfolded_model,
        save_every_n_iter=50,  # Upload rsults to wandb every 50 training iter
        training_metrics=[dinv.loss.PSNR()],
        physics = Joint_operator,  # Joint forward operators 
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

    # Pretrain time embedding
    if len(time_embed_params)>0:
        model = trainer.train(epochs=1)
    ## Add remaining param groups and train remaining epochs
    trainer.optimizer.add_param_group(unrol_param_group)
    trainer.optimizer.add_param_group(UNET_param_group)
    model = trainer.train(epochs=EPOCHS)
    return model


if __name__ == "__main__":
    train()