import os 
import numpy as np
import math

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing import Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch.autograd as autograd
from torch.optim.swa_utils import AveragedModel, SWALR, get_ema_multi_avg_fn

from metrics.metrics import Metrics

from torch.nn import functional as F
from pytorch_lightning.loggers import WandbLogger

import utils as utils
import physics as phy
import models as models
from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep
from .hqs_modules import HQS_models
from DPIR.dpir_models import DPIR_deb
from Conditioned_GAN.models.discriminator import DiscriminatorModel


class UnfoldedModel(pl.LightningModule):
    def __init__(
            self, 
            physic: Union[phy.Deblurring, phy.Inpainting, phy.SuperResolution, phy.CT], 
            kernels:phy.Kernels, 
            num_pgus:int, 
            args, 
            max_unfoled_iter:int=3, 
            wandb_logger:WandbLogger = None,
            )->None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.args = args
        self.num_gpus = num_pgus
        self.max_unfolded_iter = max_unfoled_iter
        self.kernels = kernels
        self.physic = physic
        
        self.diffusion_scheduler_ = lambda device: DiffusionScheduler(
            device=device,
            noise_schedule_type=args.noise_schedule_type 
        )
        
        # Unet 
        self.model = models.adapter_lora_model(self.args)
        
        # ---> EMA
        # self.ema_decay = 0.999
        # self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
        # for param in self.ema_model.parameters():
        #     param.requires_grad = False
            
        # Discriminator
        self.discriminator = DiscriminatorModel(im_size=self.args.data.im_size, in_channels=self.args.data.in_channels)
        
        # Disable automatic optimization
        self.automatic_optimization = False
        
        # ---> HQS model
        self.denoising_timestep_ = lambda device: GetDenoisingTimestep(device)
        self.hqs_module_ = lambda device: HQS_models(
            self.model,
            self.physic, 
            self.diffusion_scheduler_(device), 
            self.denoising_timestep_(device),
            self.args
        )
        # ---> physics
        if self.args.task=="sr" or self.args.task=="deblur":
            self.blur = self.kernels.get_blur()
        elif self.args.task=="inp":
            self.blur=self.physic.Mask
        elif self.args.task=="ct":
            self.blur = None
        else:
            raise ValueError("Task not implemented !!!")
        
        # ---> DPIR
        self.dpir_model_ = lambda device: DPIR_deb(device = device, model_name = self.args.dpir.model_name, pretrained_pth=self.args.dpir.pretrained_pth)

        # ---> parameters
        self.std_mult = 1
        self.is_good_model = 0
        self.metrics_ = lambda device: Metrics(device)
        self.mseLoss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss() #nn.BCELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg') # Squeeze
        self.nor_ = 0.0

    def _to_device(self):
        self.metrics = self.metrics_(self.device)    
        self.dpir_model = self.dpir_model_(self.device) 
        self.hqs_module = self.hqs_module_(self.device) 
        self.diffusion_scheduler = self.diffusion_scheduler_(self.device) 
        self.denoising_timestep = self.denoising_timestep_(self.device) 
        
        
    if True: 
        ## ====> Gradient penality
        def compute_gradient_penalty(self, real_samples, fake_samples, y):
            """Calculates the gradient penalty loss for WGAN GP"""
            Tensor = torch.FloatTensor
            # Random weight term for interpolation between real and fake samples
            alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
            # Get random interpolation between real and fake samples
            interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
            d_interpolates = self.discriminator(input=interpolates, y=y)
            real = Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)

            # Get gradient w.r.t. interpolates
            gradients = autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=real,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            return gradient_penalty
        
        # --- Forward pass
        def forward(self, obs, x_init, t):
            out = self.hqs_module.run_unfolded_loop( obs, x_init, t, max_iter = self.max_unfolded_iter, lambda_sig = self.args.lambda_)
            return out
        
        ## ====> Adversarial loss function: D
        # def adversarial_loss_discriminator(self, fake_pred, real_pred):
        #     adversarial_loss = nn.BCEWithLogitsLoss()#nn.BCELoss()
        #     out = adversarial_loss(fake_pred, real_pred) #
        #     # out = fake_pred.mean() + real_pred.mean()
        #     return out

        ## ====> Adversarial loss function: Diffusion model
        def adversarial_loss_diffusion(self, y, gens):
            out = self.discriminator(input=gens, y=y)
            adv_weight = 1e-5
            if self.current_epoch <= 4:
                adv_weight = 1e-2
            elif self.current_epoch <= 22:
                adv_weight = 1e-4
            return adv_weight * out.mean()
        
        # def adversarial_loss_diffusion_old(self, y, gens):
        #     fake_pred = torch.zeros(size=(y.shape[0], self.args.train.num_z_train), device=self.device)
        #     for k in range(y.shape[0]):
        #         cond = torch.zeros(1, gens.shape[2], gens.shape[3], gens.shape[4], device=self.device)
        #         cond[0, :, :, :] = y[k, :, :, :]
        #         cond = cond.repeat(self.args.train.num_z_train, 1, 1, 1)
        #         temp = self.discriminator(input=gens[k], y=cond)
        #         fake_pred[k] = temp[:, 0]

        #     gen_pred_loss = torch.mean(fake_pred[0])
        #     for k in range(y.shape[0] - 1):
        #         gen_pred_loss += torch.mean(fake_pred[k + 1])

        #     adv_weight = 1e-5
        #     if self.current_epoch <= 4:
        #         adv_weight = 1e-1
        #     elif self.current_epoch <= 22:
        #         adv_weight = 1e-4
        #     return - adv_weight * gen_pred_loss.mean()

        ## ====> L1 std penality function
        def l1_std_p(self, xest, gens, x):
            return F.l1_loss(xest, x) #- self.std_mult * np.sqrt(2 / (np.pi * self.args.train.num_z_train * (self.args.train.num_z_train + 1))) * torch.std(gens, dim=1).mean()
        
        ## ====> Gradient penality function
        def gradient_penalty(self, x_hat, x, y):
            gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)
            return self.args.train.gp_weight * gradient_penalty
        
        ## ====> Drift penality function
        # def drift_penalty(self, real_pred):
        #     return 0.001 * torch.mean(real_pred ** 2)
        
    ## ====> training penality function
    def training_step(self, batch, batch_idx):
        self.model.train()
        self.discriminator.train()
        if 0:
            self.diagnos_model()
            sedrf
        
        
        # Device
        self._to_device()
        
        # image 
        B,C = batch.shape[:2]
        
        # ---> sample noisy x_t
        t = self.diffusion_scheduler.sample_times(B)
        noise_target = torch.randn_like(batch)
        
        x_t = self.diffusion_scheduler.sample_xt(batch, t, noise=noise_target)
        
        # ---> observation y 
        if self.args.task == "deblur":
            y = self.physic.y(batch, self.blur)
        elif self.args.task=="sr":
            y = self.physic.y(batch, self.blur, self.args.physic.sr.sf)
        elif self.args.task=="inp":
            y = self.physic.y(batch)
        elif self.args.task=="ct":
            y = self.physic.y(batch)
        else:
            raise ValueError(f"This task ({self.args.task})is not implemented!!!")
        # ---> initialisation unfolded model
        if self.args.dpir.use_dpir and self.args.task == "deblur":
            y_ = utils.inverse_image_transform(y)
            x_init = self.dpir_model.run(y_, self.blur, iter_num = 1) #y.clone()
            x_init = utils.image_transform(x_init)
        elif self.args.task=="sr":
            with torch.no_grad():
                y_ = self.physic.upsample(utils.inverse_image_transform(y))
                if self.args.dpir.use_dpir:
                    x_init = self.dpir_model.run(y_, self.blur, iter_num = 1)
                x_init = utils.image_transform(y_)
        elif self.args.task=="inp":
            x_init = y.clone()
        elif self.args.task=="ct":
            x_init = self.physic.init(y)
        else:
            raise ValueError(f"This task ({self.args.task})is not implemented!!!")
            
        # ---> unfolded model
        obs = {"x_t":x_t, "y":y, "b":batch}
        if self.args.task not in ["inp", "ct"]:
            obs["blur"] = self.blur
        
        # ---> optimizers
        opt_g, opt_d = self.optimizers()
        
        self.get_trainable_params()
        
        # ---> train generator
        valid = torch.FloatTensor(B, 1).fill_(1.0).requires_grad_(False).to(self.device)
        fake = torch.FloatTensor(B, 1).fill_(0.0).requires_grad_(False).to(self.device)
        
        gens = torch.zeros(
            size=(y.size(0), 
                self.args.train.num_z_train, 
                self.args.data.in_channels, 
                self.args.data.im_size, 
                self.args.data.im_size),
            device=self.device
        )
        if self.args.train.num_z_train>1:
            for id_z in range(self.args.train.num_z_train):
                out = self.forward(obs, x_init, t)
                gens[:, id_z, :, :, :] = out["xstart_pred"]
            xest = torch.mean(gens, dim=1)
        else:   
            out = self.forward(obs, x_init, t)
            xest = out["xstart_pred"]
        xest = xest.mul(2).add(-1)
        
        # ---> adversarial loss is binary cross-entropy
        if self.args.task=="sr":
            y_up = self.physic.upsample(y)
        elif self.args.task=="ct":
            y_up = self.physic.ATx(y)
        else:
            y_up=y.clone()
            
        if self.args.train.num_z_train>1:
            g_loss = self.adversarial_loss_diffusion(y_up, gens)
        else:
            d_out = self.discriminator(input=xest, y=y_up)
            g_loss = self.adversarial_loss(d_out, valid)
        g_loss += self.mseLoss(out["eps_pred"], noise_target).mean()
        g_loss += self.args.train.scale_lpips*self.lpips(xest.add(1).div(2), batch.add(1).div(2)).mean()
        self.log('g_loss', g_loss, prog_bar=True)
        
        # ---> update
        self.manual_backward(g_loss)
        
        # clip gradients
        # if self.args.train.gradient_clip:
        #     self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        
        # accumulate gradients of N batches
        if self.args.train.gradient_accumulation and (batch_idx + 1) % self.args.train.acc_grad_step == 0:
            opt_g.step()
            opt_g.zero_grad()
        elif not self.args.train.gradient_accumulation:
            opt_g.step()
            opt_g.zero_grad()
        else:
            pass

        # --- > train discriminator
        out = self.forward(obs, x_init, t)
        x_hat = out["xstart_pred"]
        x_hat = x_hat.mul(2).add(-1)
        
        real_pred = self.discriminator(input=batch, y=y_up)
        fake_pred = self.discriminator(input=x_hat, y=y_up)

        d_loss = self.adversarial_loss(real_pred, valid).mean()/2    
        d_loss += self.adversarial_loss(fake_pred, fake).mean()/2 
        d_loss += self.args.train.gradient_scale_dist*self.gradient_penalty(x_hat, batch, y_up)
        
        self.manual_backward(d_loss)  
        # clip gradients
        if self.args.train.gradient_clip:
            self.clip_gradients(opt_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            
        # accumulate gradients of N batches
        if self.args.train.gradient_accumulation and (batch_idx + 1) % self.args.train.acc_grad_step == 0:
            opt_d.step()
            opt_d.zero_grad()
        else:
            opt_d.step()
            opt_d.zero_grad()  
        
        # --- >> 
        nor_new = self.parameters_changes(self.model).detach().cpu()
        err = torch.abs(self.nor_ - nor_new)
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('norm params', nor_new, prog_bar=True)
        self.log('rel. error', err, prog_bar=True)
        self.nor_ = nor_new
        
    # # --- >> EMA update    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     """Update EMA after each batch"""
    #     self.ema_model.update_parameters(self.model)   
    
    # # --- >> Update function   
    # def ema_update(self, avg_model_param, model_param, num_averaged):
    #     """Custom EMA update function"""
    #     print(avg_model_param)
    #     avg_model_param = torch.tensor(avg_model_param) if isinstance(avg_model_param, list) else avg_model_param
    #     model_param = torch.tensor(model_param) if isinstance(model_param, list) else model_param

    #     return self.ema_decay * avg_model_param + (1 - self.ema_decay) * model_param.detach().cpu()
    
    # --- >> Validation function           
    def validation_step(self, batch, batch_idx, external_test=False):
       
        if batch_idx==1:
            # image 
            self.discriminator.eval()
            
            # Device
            self._to_device()
            
            # image 
            B,C = batch.shape[:2]
            # ---> sample noisy x_t
            t = self.diffusion_scheduler.sample_times(B)
            noise_target = torch.randn_like(batch)
            x_t = self.diffusion_scheduler.sample_xt(batch, t, noise=noise_target)
            
            # ---> observation y 
            if self.args.task == "deblur":
                y = self.physic.y(batch, self.blur)
            elif self.args.task=="sr":
                y = self.physic.y(batch, self.blur, self.args.physic.sr.sf)
            elif self.args.task=="inp":
                y = self.physic.y(batch)
            elif self.args.task=="ct":
                y = self.physic.y(batch)
            else:
                raise ValueError(f"This task ({self.args.task})is not implemented!!!")
        
            # ---> initialisation unfolded model
            if self.args.dpir.use_dpir and self.args.task == "deblur":
                if self.args.dpir.use_dpir:
                    y_ = utils.inverse_image_transform(y)
                    x_init = self.dpir_model.run(y_, self.blur, iter_num = 1) #y.clone()
                    x_init = utils.image_transform(x_init)
                else:
                    x_init = y.clone()
            elif self.args.task=="sr":
                with torch.no_grad():
                    y_ = self.physic.upsample(utils.inverse_image_transform(y))
                    if self.args.dpir.use_dpir:
                        x_init = self.dpir_model.run(y_, self.blur, iter_num = 1)
                    x_init = utils.image_transform(y_)
            elif self.args.task=="inp":
                x_init = y.clone()
            elif self.args.task=="ct":
                x_init = self.physic.init(y)
            else:
                raise ValueError(f"This task ({self.args.task})is not implemented!!!")
            
            # ---> unfolded model
            obs = {"x_t":x_t, "y":y, "b":batch}
            if self.args.task not in ["inp", "ct"]:
                obs["blur"] = self.blur
            
            
            gens = torch.zeros(size=
                        (y.size(0), 
                        self.args.train.num_z_test, 
                        self.args.data.in_channels, 
                        self.args.data.im_size, 
                        self.args.data.im_size),
                device=self.device
            )
            
            if self.args.train.num_z_test>1:
                for id_z in range(self.args.train.num_z_test):
                    out = self.forward(obs, x_init, t)
                    gens[:, id_z, :, :, :] = out["xstart_pred"]
                xest = torch.mean(gens, dim=1)
            else:   
                out = self.forward(obs, x_init, t)
                xest = out["xstart_pred"]

            xest = torch.mean(gens, dim=1)
            
            # ---> adversarial loss is binary cross-entropy
            valid = torch.FloatTensor(B, 1).fill_(1.0).requires_grad_(False).to(self.device)   
            if self.args.task=="sr":     
                y_up = self.physic.upsample(y)
            elif self.args.task=="ct":
                y_up = self.physic.ATx(y)
            else:
                y_up=y.clone()
             
            if self.args.train.num_z_train>1:
                g_loss_valid = self.adversarial_loss_diffusion(y_up, gens)
                g_loss_valid += self.l1_std_p(xest, gens, batch)
            else:
                g_loss_valid= 0
            g_loss_valid += self.mseLoss(xest, batch).mean()
            
            self.log('test -- g_loss', g_loss_valid, prog_bar=True)
            self.log("mse", self.metrics.mse_function(out["xstart_pred"].detach().cpu(), batch.add(1).div(2).detach().cpu()).item(), prog_bar=True)
            self.log("psnr", self.metrics.psnr_function(out["xstart_pred"].detach().cpu(), batch.add(1).div(2).detach().cpu()).item(), prog_bar=True)
            
            self.wandb_logger.log_image(
                    key = "training",
                    images = [
                        utils.get_batch_rgb_from_batch_tensor(out["xstart_pred"]), 
                        utils.get_batch_rgb_from_batch_tensor(batch.add(1).div(2).detach()), 
                        utils.get_batch_rgb_from_batch_tensor(y), 
                        utils.get_batch_rgb_from_batch_tensor(x_t), 
                        utils.get_batch_rgb_from_batch_tensor(out["z_input"]), 
                        utils.get_batch_rgb_from_batch_tensor(out["aux"])
                    ],
                    caption=["x0_hat", "x0", "y", "xt", "z_input", "z"]
                )
        else:
            pass
        
    def configure_optimizers(self):
        # ---> leanable parameters
        self.model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.discriminator_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        
        # ---> optimizers        
        opt_g = torch.optim.AdamW(self.model_params, lr=self.args.train.learning_rate, weight_decay=0.01)
        opt_d = torch.optim.AdamW(self.discriminator_params, lr=self.args.train.learning_rate, weight_decay=0.01)
    
        return [opt_g, opt_d] 
    
    # --- >> Save checkpoints 
    def on_save_checkpoint(self, checkpoint):
        
        #self.current_epoch      
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model
        
        # Get trainable parameter names
        trainable_params = {k for k, v in self.model.named_parameters() if v.requires_grad}
        checkpoint['state_dict'] = {k: v for k, v in self.model.state_dict().items() if k in trainable_params}

        print("Saving only trainable parameters!")
        
    # --- >> Load checkpoints
    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Loaded only trainable parameters!")
    
    # --- >> Get parameters  
    def get_trainable_params(self):
        n_layers,m_layers,T_params,T_frozen=0,0,0,0
        for name, params in self.model.named_parameters():

            if "lora" in name and params.requires_grad:
                n_layers += 1
                T_params += params.numel()
            elif "lora" in name and not params.requires_grad:
                T_frozen += params.numel()
                m_layers += 1
                
        self.log(f"trained p:", T_params, prog_bar=True)
        self.log(f"trained l:", n_layers, prog_bar=True)
        
    def parameters_changes(self, model):
        norm = 0.0
        for name, params in model.named_parameters():
            if params.requires_grad:
                nn = params.numel()
                norm+= params.sum()/nn
        return norm
    
    def diagnos_model(self):
        import sys
        for name, params in self.model.named_parameters():
            if "lora" in name and params.requires_grad:
                self.log("****** lora with required grad", params.requires_grad, prog_bar=True)
                print("****** lora with required grad", params.requires_grad)
            elif "lora" in name and not params.requires_grad:
                self.log("------ lora with not required grad", params.requires_grad, prog_bar=True)
                print("------ lora with not required grad", params.requires_grad)
         
        sys.exit()
        
        
## --- >> Checkpoint saving
class SaveModelrWithCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module: UnfoldedModel, checkpoint):
        """Save model and EMA model parameters"""
        checkpoint["model_state_dict"] = {
            k: v for k, v in pl_module.model.state_dict().items()
            if pl_module.model.state_dict()[k].requires_grad
        }
        # if hasattr(pl_module, "ema_model"):
        #     checkpoint["ema_model_state_dict"] = {
        #         k:pl_module.ema_model.state_dict()[k] for k,v in pl_module.model.state_dict().items() 
        #         if k!="n_averaged" and pl_module.model.state_dict()[k].required_grad
        #     }
            
        #     checkpoint["n_averaged"] = pl_module.ema_model.state_dict["n_averaged"]
        

    def on_load_checkpoint(self, trainer, pl_module: UnfoldedModel, checkpoint):
        """Load model and EMA generator parameters"""
        pl_module.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        # if hasattr(pl_module, "ema_model"):
        #     pl_module.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)