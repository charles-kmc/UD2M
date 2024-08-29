import deepinv as dinv
from .trainer import TrainerPlt
from torch.optim import Adam, lr_scheduler
from torch.nn import Module
from torch import load
import wandb

class forward_denoiser(Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser 
    
    def forward(self, y, physics, **kwargs):
        sigma = physics.sigma if hasattr(physics, "sigma") else None 
        return self.denoiser(y, sigma = sigma)

def train_gaussian_denoiser(
        denoiser: Module,
        train_loader,
        test_loader,
        epochs = 20,
        optimizer = None,
        scheduler = None,
        wandb_config = None,
        sigma_min = 0.,
        sigma_max = 0.5,
        save_path='models/saved',
        lr=5e-4,
        physics = None,
        name = None
):
    if physics is None:
        physics = dinv.physics.UniformGaussianNoise(sigma_min=sigma_min,sigma_max=sigma_max)

    if not isinstance(denoiser, forward_denoiser):
        denoiser = forward_denoiser(denoiser=denoiser)

    if optimizer is None:
        optimizer=Adam(
            params=denoiser.parameters(),
            lr=lr,
        )
    if scheduler is None:
        scheduler = lr_scheduler.StepLR(optimizer,step_size=epochs//10 + 1, gamma=0.5)
    name = (type(denoiser.denoiser).__name__ if name is None else name) + '_denoiser_' + type(train_loader.dataset).__name__
    if save_path is not None:
        import os 
        save_path = os.path.join(
            save_path,
            name
        )

        if os.path.exists(save_path):
            try:
                ckp = load(save_path + '/ckp.pth.tar')
                denoiser.load_state_dict(ckp["state_dict"])
                if ckp["epoch"] >= epochs-1.5:
                    return denoiser.denoiser
                else: 
                    epochs = epochs-ckp["epoch"]
                    print(f"Training remaining {epochs} epochs")

            except:
                print("Could not load pretrained model: Training New Denoiser")
    wandb_config_new = wandb_config.copy()
    wandb_config_new["name"] = name
    trainer = TrainerPlt(
        denoiser,
        save_every_n_iter = 100,
        training_metrics = [dinv.loss.PSNR()],
        physics=physics,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        epochs=epochs,
        save_path=save_path,
        verbose=1,
        show_progress_bar=True,
        plot_images=True,
        plot_metrics=True,
        online_measurements=True,
        wandb_vis=save_path is not None,
        wandb_setup=wandb_config_new,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.save_path = save_path

    denoiser = trainer.train()

    if wandb.run is not None:
        wandb.finish()
    
    return denoiser.denoiser
    