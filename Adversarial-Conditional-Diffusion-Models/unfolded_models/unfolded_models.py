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
import deepinv as dinv
import utils as utils
import physics as phy
import models as models
from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep
from .hqs_modules import HQS_models, DDNoise, StackedJointPhysics
from DPIR.dpir_models import DPIR_deb
from Conditioned_GAN.models.discriminator import DiscriminatorModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class UnfoldedModel(pl.LightningModule):
    def __init__(
            self, 
            physic: Union[phy.Deblurring, phy.Inpainting, phy.SuperResolution, phy.CT], 
            kernels:phy.Kernels, 
            num_pgus:int, 
            args, 
            device,
            max_unfoled_iter:int=3, 
            wandb_logger:WandbLogger = None,
            dphys = None,
            learn_RAM = True,
            PnP_Forward = False,
        )->None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.args = args
        self.num_gpus = num_pgus
        self.max_unfolded_iter = max_unfoled_iter
        self.kernels = kernels
        self.physic = physic
        self.d_phys = dphys
        self.PnP_Forward = PnP_Forward

        # ---> physics
        if self.args.task in ["sr" ,"deblur"]:
            self.blur = self.kernels.get_blur()
            print(f"Blur kernel shape: {self.blur.shape}")
        elif self.args.task=="inp":
            self.blur=self.physic.Mask
        elif self.args.task in ["ct", "phase_retrieval", "jpeg"]:
            self.blur = None
        else:
            raise ValueError("Task not implemented !!!")
        
        # ---> DPIR
        if self.args.dpir.use_dpir:
            self.dpir_model = DPIR_deb(
                device=device, 
                model_name=self.args.dpir.model_name, 
                pretrained_pth=self.args.dpir.pretrained_pth
            )

        # ---> parameters
        self.std_mult = 1
        self.is_good_model = 0
        self.metrics = Metrics(device)
        self.mseLoss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

        self.diffusion_scheduler = DiffusionScheduler(
                device=device,
                noise_schedule_type=args.noise_schedule_type 
            )
        
        # ---> Unet with LoRA and discriminator
        self.model = models.adapter_lora_model(self.args)
        self.discriminator = DiscriminatorModel(
            im_size=self.args.data.im_size, 
            in_channels=self.args.data.in_channels
        )

        print(f"The rank of LoRA is {self.args.model.rank}")
        print(f"Number of parameters in the model: {self._total_params()}")
        print(f"Number of trainable parameters in the discriminator: {self._total_params_dis()}")
        print(f"Number of trainable parameters in the model: {self._trained_params()}")
        print(f"Ratio of trainable parameters in model: {self._trained_params()/self._total_params()*100:.2f}%")
        print(f"Ratio of total trainable parameters: {(self._total_params_dis() + self._trained_params())/(self._total_params_dis() + self._total_params())*100:.2f}%")
        print(f"Total number of parameters: {self._total_params_dis() + self._total_params()}")

        
        self.denoising_timestep = GetDenoisingTimestep(
            self.diffusion_scheduler,
            device
        )
        
        # ---> RAM
        if self.d_phys is not None:
            from ram import RAM
            self.RAM = RAM(device=device)
            #count the number of parameters in RAM
            print(f"Number of parameters in RAM: {sum(p.numel() for p in self.RAM.parameters())}")
            if learn_RAM:
                self.RAM.train()
                # for k, v in self.RAM.named_parameters():
                #     v.requires_grad = True
            else:
                self.RAM.eval()
                for k, v in self.RAM.named_parameters():
                    v.requires_grad = False
            self.stacked_physic = StackedJointPhysics(
                self.d_phys, self.diffusion_scheduler
            )
        # ---> HQS model
        self.hqs_module = HQS_models(
            self.model,
            self.physic, 
            self.diffusion_scheduler, 
            self.denoising_timestep,
            self.args,
            device,
            max_unfolded_iter=self.max_unfolded_iter,
            RAM = self.RAM if PnP_Forward else None,
            RAM_Physics = dphys if PnP_Forward else None,
        )
    
        self.psnr_outputs = []
        self.hqs_module = self.hqs_module.to(device) 
        
        # ---> Disable automatic optimization
        self.automatic_optimization = False
    def _total_params_dis(self):
        """Returns the total number of parameters in the discriminator"""
        return sum(p.numel() for p in self.discriminator.parameters())
    def _trained_params(self):
        """Returns the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    def _total_params(self):
        """Returns the total number of parameters in the model"""
        return sum(p.numel() for p in self.model.parameters())

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
        gradient_penalty = ((gradients.norm(2, dim=1)) ** 2).mean() 
        return gradient_penalty
    
    # --- Forward pass
    def forward(self, obs, x_init, t, collect_ims = False):
        out = self.hqs_module.run_unfolded_loop(obs, x_init, t, max_iter = self.max_unfolded_iter, lambda_sig = self.args.lambda_, collect_ims = collect_ims)
        return out

    ## ====> Adversarial loss function: Diffusion model
    def adversarial_loss_diffusion(self, y, gens):
        out = self.discriminator(input=gens, y=y)
        adv_weight = 1e-5
        if self.current_epoch <= 4:
            adv_weight = 1e-2
        elif self.current_epoch <= 22:
            adv_weight = 1e-4
        return adv_weight * out.mean()

    ## ====> L1 std penality function
    def l1_std_p(self, xest, gens, x):
        return F.l1_loss(xest, x) 
    
    ## ====> Gradient penality function
    def gradient_penalty(self, x_hat, x, y):
        gradient_penalty = self.compute_gradient_penalty(x.data, x_hat.data, y.data)
        return self.args.train.gp_weight * gradient_penalty
    
    def wgan_gp_critic_loss(self, real_data, fake_data, y, valid, fake):
        real_validity = self.discriminator(input=real_data, y=y)
        fake_validity = self.discriminator(input=fake_data, y=y)
        real_validity = self.adversarial_loss(real_validity, valid).mean()
        fake_validity = self.adversarial_loss(fake_validity, fake).mean()
        gp = self.gradient_penalty(fake_data, real_data, y)

        # Compute WGAN GP loss
        wgan_gp_loss = (torch.mean(real_validity) + torch.mean(fake_validity))/2

        # Compute gradient penalty
        wgan_gp_loss += gp

        return wgan_gp_loss
    def _proc_batch(self, batch):
        if isinstance(batch, list):
            if isinstance(batch[1], dict) and "y_label" in batch[1].keys():
                batch, y_label = batch[0], batch[1]["y_label"]
            else:
                raise ValueError("Batch should either be a tensor of size (B,C,H,W) or a tuple of (tensor, dict) where dict contains class conditional y")
        else: 
            y_label = None
        return batch, y_label
    
    def RAM_Initialization(self, y, xt, t):
        if self.args.use_RAM:
            if isinstance(t, torch.Tensor) and len(t)>1:
                t = int(t[0].item())
            self.stacked_physic.set_t(t)
            xt_init = (xt.clone() + self.stacked_physic.physics_list[1].alphas_cumprod[t].sqrt())/2 
            min = torch.min(torch.tensor([p.noise_model.sigma for p in self.stacked_physic.physics_list]))
            y_ram = dinv.utils.TensorList([
                            utils.inverse_image_transform(y)*min/self.d_phys.noise_model.sigma, 
                            xt_init*min/self.stacked_physic.physics_list[1].noise_model.sigma
                        ])
            x_init = self.RAM(y_ram, self.stacked_physic)
            return utils.image_transform(x_init)
        else:
            return y
        
        
    ## ====> training penality function
    def training_step(self, batch, batch_idx):
        self.model.train()
        self.discriminator.train()
        batch, y_label = self._proc_batch(batch)
        # Device
        # self._to_device()
        
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
        else:
            raise ValueError(f"This task ({self.args.task})is not implemented!!!")
        # ---> initialisation unfolded model
        if self.args.task == "deblur":
            if self.args.dpir.use_dpir: 
                x_init = self.dpir_model.run(utils.inverse_image_transform(y), self.blur, iter_num=1)
                x_init = utils.image_transform(x_init)
            elif self.d_phys is not None:
                with torch.no_grad() if batch_idx % 2 == 1 else torch.enable_grad():
                    x_init = self.RAM_Initialization(y, x_t, t)
            else:
                x_init = y.clone()
                
        elif self.args.task=="sr":
            with torch.no_grad() if batch_idx % 2 == 1 else torch.enable_grad():
                if self.d_phys is not None:
                    print(f"Allocated memory RAM before: {torch.cuda.memory_allocated()  / 1024**3:.2f} GB")
                    x_init = self.RAM_Initialization(y, x_t, t)
                    print(f"Allocated memory RAM after: {torch.cuda.memory_allocated()  / 1024**3:.2f} GB")

                elif self.args.dpir.use_dpir :
                    x_init = self.dpir_model.run(utils.inverse_image_transform(y), self.blur, iter_num=1) 
                    x_init = utils.image_transform(x_init)
                else:
                    x_init = y.clone()
        elif self.args.task=="inp":
            # with torch.no_grad():
            if self.d_phys is not None:
                with torch.no_grad() if batch_idx % 2 == 1 else torch.enable_grad():
                    x_init = self.RAM_Initialization(y, x_t, t)
            else:
                x_init = y.clone()
        else:
            raise ValueError(f"This task ({self.args.task})is not implemented!!!")
            
        # ---> unfolded model
        obs = {"x_t":x_t, "y":y, "b":batch, "y_label":y_label}
        if self.args.task not in ["inp", "ct"]:
            obs["blur"] = self.blur
        
        # ---> optimizers
        opt_g, opt_d = self.optimizers()
        opt_g.zero_grad()
        opt_d.zero_grad()
        self.get_trainable_params()
        
        # ---> train generator
        valid = torch.FloatTensor(B, 1).fill_(1.0).requires_grad_(False).to(self.device)
        fake = torch.FloatTensor(B, 1).fill_(0.0).requires_grad_(False).to(self.device)
 
        out = self.forward(obs, x_init, t)
        xest = out["xstart_pred"].mul(2).add(-1)
        
        # ---> adversarial loss is binary cross-entropy
        if self.args.task=="sr":
            y_up = self.physic.upsample(y)
        else:
            y_up=y.clone()
        
        if batch_idx % 2 == 0:
            d_out = self.discriminator(input=xest, y=y_up)
            g_loss = self.adversarial_loss(d_out, valid)
            g_loss += self.mseLoss(out["eps_pred"], noise_target).mean()
            g_loss += self.args.train.scale_lpips*self.lpips(xest.add(1).div(2), batch.add(1).div(2)).mean()
            self.log('g_loss', g_loss, prog_bar=True)
            for i in range(self.hqs_module.num_weights_hqs):
                self.log(f"SP_param_{i}", self.hqs_module.SP_Param[i].exp(), prog_bar=False)
                self.log(f"LambdaSIG_param_{i}", self.hqs_module.LambdaSIG[i].exp(), prog_bar=False)
                self.log(f"TAUG_param_{i}", 1000*self.hqs_module.TAUG[i], prog_bar=False)  # T_ are multiplied by 1000 in script to match size of noise schedule
                self.log(f"TLAT_param_{i}", 1000*self.hqs_module.TLAT[i], prog_bar=False)
            
            # ---> update
            self.manual_backward(g_loss)
            opt_g.step()

        # --- > train discriminator
        if batch_idx % 2 == 1:     
            # out = self.forward(obs, x_init, t)
            # xest = out["xstart_pred"].mul(2).add(-1)
            real_pred = self.discriminator(input=batch, y=y_up)
            fake_pred = self.discriminator(input=xest, y=y_up)
            d_loss = self.adversarial_loss(real_pred, valid).mean()/2    
            d_loss += self.adversarial_loss(fake_pred, fake).mean()/2 
            d_loss += self.gradient_penalty(xest, batch, y_up)
            
            self.manual_backward(d_loss) 
            opt_d.step()
            
            # --- >> 
            self.log('d_loss', d_loss, prog_bar=True)

        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak Allocated Memory: {peak_memory_gb:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated()  / 1024**3:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved()  / 1024**3:.2f} GB")
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        
        dsfatrwrgfs
            
    def print_memory(self, tag=""):
        allocated = torch.cuda.memory_allocated() / 1024**3  # in GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        
        
    # --- >> Validation function           
    def validation_step(self, batch, batch_idx, external_test=False):
        # image 
        self.discriminator.eval()
        batch, y_label = self._proc_batch(batch)
        
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
        with torch.no_grad():
            if self.d_phys is not None:
                x_init = self.RAM_Initialization(y, x_t, t)
            elif self.args.task == "deblur":
                if self.args.dpir.use_dpir:
                    x_init = self.dpir_model.run(utils.inverse_image_transform(y), self.blur, iter_num = 1) 
                    x_init = utils.image_transform(x_init)
                else:
                    x_init = y.clone()
            elif self.args.task=="sr":
                x_init = utils.image_transform(y_)
            elif self.args.task=="inp":
                x_init = y.clone()
            else:
                raise ValueError(f"This task ({self.args.task})is not implemented!!!")
        
        # ---> unfolded model
        obs = {"x_t":x_t, "y":y, "b":batch, "y_label":y_label}
        if self.args.task not in ["inp", "ct"]:
            obs["blur"] = self.blur
        
        out = self.forward(obs, x_init, t, collect_ims=batch_idx==0)
        xest = out["xstart_pred"].detach()
        
        self.log("mse", self.metrics.mse_function(xest, batch.add(1).div(2)).item(), prog_bar=True)
        self.log("psnr", self.metrics.psnr_function(xest, batch.add(1).div(2)).item(), prog_bar=True)
        self.psnr_outputs.append(self.metrics.psnr_function(xest, batch.add(1).div(2)))
        
        if batch_idx == 0:
            self.wandb_logger.log_image(
                    key = "training",
                    images = [
                        utils.get_batch_rgb_from_batch_tensor(xest), 
                        utils.get_batch_rgb_from_batch_tensor(batch.add(1).div(2).detach()), 
                        utils.get_batch_rgb_from_batch_tensor(y.add(1).div(2).detach()), 
                        utils.get_batch_rgb_from_batch_tensor(x_t), 
                        utils.get_batch_rgb_from_batch_tensor(out["z_input"]), 
                        utils.get_batch_rgb_from_batch_tensor(out["aux"])
                    ],
                    caption=["x0_hat", "x0", "y", "xt", "z_input", "z"]
                )
            self.wandb_logger.log_image(
                key = "Unfolded_Loop",
                images = [utils.get_batch_rgb_from_batch_tensor(x_init.add(1).div(2).detach())] + [
                    utils.get_batch_rgb_from_batch_tensor(v.add(1).div(2).detach()) for k, v in out["IMDICT"].items()
                ],
                caption=["x_init"] + [f"{k}" for k in out["IMDICT"].keys()]
            )
    
    def on_validation_epoch_end(self):
        psnrs = torch.stack([torch.tensor(p) for p in self.psnr_outputs])
        psnr_mean = torch.mean(psnrs)
        self.log("psnr_mean", psnr_mean, prog_bar=True)
        self.psnr_outputs.clear()

    def configure_optimizers(self):
        # ---> leanable parameters
        self.model_params = list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(filter(lambda p: p.requires_grad, self.hqs_module.parameters())) 
        self.RAM_params = list(filter(lambda p: p.requires_grad, self.RAM.parameters())) if not self.PnP_Forward else []
        self.discriminator_params = list(filter(lambda p: p.requires_grad, self.discriminator.parameters()))
        
        # ---> optimizers        
        opt_g = torch.optim.AdamW([
                {"params":self.model_params},
                {"params":self.RAM_params, "lr":self.args.train.learning_rate*0.1},  # Use smaller learning rate for RAM to avoid overfitting
            ], lr=self.args.train.learning_rate, weight_decay=0.01)
        opt_d = torch.optim.AdamW(self.discriminator_params, lr=self.args.train.learning_rate, weight_decay=0.01)
    
        return [opt_g, opt_d] 
    
    # --- >> Save checkpoints 
    def on_save_checkpoint(self, checkpoint):
        
        #self.current_epoch      
        checkpoint["beta_std"] = self.std_mult
        checkpoint["is_valid"] = self.is_good_model
        
        # Get trainable parameter names
        trainable_params = {k for k, v in self.model.named_parameters() if v.requires_grad}
        trainable_hqs_params = {k for k, v in self.hqs_module.named_parameters() if v.requires_grad}
        trainable_ram_params = {k for k, v in self.RAM.named_parameters() if v.requires_grad}
        checkpoint['state_dict'] = {k: v for k, v in self.model.state_dict().items() if k in trainable_params}
        checkpoint['HQS_state_dict'] = {k: v for k, v in self.hqs_module.state_dict().items() if k in trainable_hqs_params}
        checkpoint['RAM_state_dict'] = {k: v for k, v in self.RAM.state_dict().items() if k in trainable_ram_params}
        checkpoint['Discriminator_state_dict'] = self.discriminator.state_dict()
        checkpoint['opt_g_state_dict'] = self.optimizers()[0].state_dict()
        checkpoint['opt_d_state_dict'] = self.optimizers()[1].state_dict()
        print("Saving only trainable parameters!")
        
    # --- >> Load checkpoints
    def on_load_checkpoint(self, checkpoint):
        self.std_mult = checkpoint["beta_std"]
        self.is_good_model = checkpoint["is_valid"]
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.hqs_module.load_state_dict(checkpoint['HQS_state_dict'], strict=False)
        self.RAM.load_state_dict(checkpoint['RAM_state_dict'], strict=False)
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
class SaveModelWithCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module: UnfoldedModel, checkpoint):
        """Save model and EMA model parameters"""
        checkpoint["model_state_dict"] = {
            k: v for k, v in pl_module.model.state_dict().items()
            if pl_module.model.state_dict()[k].requires_grad
        }
        checkpoint["HQS_state_dict"] = {
            k: v for k, v in pl_module.hqs_module.state_dict().items()
            if pl_module.hqs_module.state_dict()[k].requires_grad
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
        pl_module.hqs_module.load_state_dict(checkpoint["HQS_state_dict"], strict=False)
        
        # if hasattr(pl_module, "ema_model"):
        #     pl_module.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)