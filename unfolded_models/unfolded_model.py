import math
import numpy as np
import random
import copy
import os
import psutil
import wandb
import time
import cv2
import blobfile as bf
import datetime
import copy
import yaml
import itertools

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
from accelerate import Accelerator
from torch.optim import AdamW
import torch
import torch.nn as nn

from .diffusion_schedulers import DiffusionScheduler, GetDenoisingTimestep
from .hqs_modules import HQS_models
from DPIR.dpir_models import DPIR_deb

from typing import Union, Tuple, List, Dict, Any, Optional
import physics as phy
import utils as utils

import WGAN as wgan

# extract into tensor       
def extract_tensor(
        arr:torch.Tensor, 
        timesteps:torch.Tensor, 
        broadcast_shape:torch.Size
    ) -> torch.Tensor:
        out = arr.to(device=arr.device)[timesteps].float()
        while len(out.shape) < len(broadcast_shape):
            out = out[..., None]
        return out.expand(broadcast_shape)

def LinearScheduler(timesteps, beta_start = 0.0001, beta_end=0.02):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
    return betas

def CosineScheduler(timesteps, beta_max =0.999):
    cos_function = lambda x: math.cos((x + 0.008) / (1.008) * math.pi / 2)**2
    betas = []
    for t in range(timesteps):
        t1 = t/timesteps
        t2 = (t+1)/timesteps
        beta = min(1 - cos_function(t2) / cos_function(t1), beta_max)
        betas.append(beta)
    return np.array(betas)

class Trainer:
    def __init__(
        self,
        model,
        diffusion_scheduler:DiffusionScheduler,
        physic:Union[phy.Deblurring, phy.Inpainting,phy.SuperResolution],
        kernels:phy.Kernels,
        hqs_module:HQS_models,
        trainloader,
        testloader,
        logger,
        device:str,
        args,
        max_unfolded_iter:int = 3,
        ) -> None:
        
        self.args = args
        self.logger = logger
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
        self.kernels = kernels
        self.max_unfolded_iter = max_unfolded_iter
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.hqs_module = hqs_module
        
        self.copy_model = copy.deepcopy(self.model)
        
        # ---> Discriminator 
        discriminator = wgan.Discriminator(self.args.im_size).to(device)
        discriminator.apply(wgan.weights_init_normal)
        self.discriminator = discriminator.to(device)
        
        # ---> set seed
        self._seed(self.args.seed)
        
        # ---> Accelerator: DDP
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        
        # ---> set training
        self._set_training()
        self.metrics = utils.Metrics(self.device)
        
    def _set_training(self):
        model_parameters(self.model, self.logger)
        self.model_params_learnable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        
        # ---> Optimizers
        b1 = 0.5
        b2 = 0.999
        self.optimizer = AdamW(self.model_params_learnable, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.learning_rate, betas=(b1, b2))
        self.optimizer_info = torch.optim.Adam(
            itertools.chain(self.model_params_learnable, self.discriminator.parameters()), lr=self.args.learning_rate, betas=(b1, b2))
        
        # ---> Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(100 * 0.2), gamma=0.1)
        
        # ---> accelerator
        self.model, self.optimizer, self.optimizer_D, self.optimizer_info, self.trainloader, self.testloader, self.scheduler = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.optimizer_D,
            self.optimizer_info,
            self.trainloader,
            self.testloader, 
            self.scheduler, 
        )
        
        # ---> HQS model
        self.denoising_timestep = GetDenoisingTimestep(self.device)
        self.hqs_model = HQS_models(
            self.model,
            self.physic, 
            self.diffusion_scheduler, 
            self.denoising_timestep,
            self.args
        )
        
        # ---> Losses 
        self.adversarial_loss = torch.nn.MSELoss().to(self.device)
        self.categorical_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.continuous_loss = torch.nn.MSELoss().to(self.device)
        # Loss weights
        self.lambda_cat = 1
        self.lambda_con = 0.1
        
        # ---> DPIR model
        self.dpir_model = DPIR_deb(device = self.device, model_name = self.args.dpir.model_name, pretrained_pth=self.args.dpir.pretrained_pth)

        # ---> lpips loss
        self.loss_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
        
        
        
    def training(self, epochs):
        """_summary_
        """
        if self.args.dpir.use_dpir:
            init = f"init_model_{self.args.dpir.model_name}"
        else:
            init = f"init_xty_{self.args.init_xt_y}"
        if self.args.use_wandb:
            formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dir_w = "/users/cmk2000/sharedscratch/Results/wandb"
            os.makedirs(dir_w, exist_ok=True)
            wandb.init(
                    # set the wandb project where this run will be logged
                    project=f"Training Unfolded Conditional Diffusion Models {self.args.task} {self.args.physic.operator_name}",
                    dir = dir_w,
                    config={
                        "consistency":self.args.use_consistancy,
                        "learning_rate": self.args.learning_rate,
                        "architecture": "Deep Unfolding",
                        "Structure": "HQS",
                        "dataset": "FFHQ - 256",
                        "epochs": epochs,
                        "dir": dir_w,
                        "pertub":self.args.pertub,
                        "max_unfolded_iter":self.args.max_unfolded_iter,
                        "Operator name": self.args.physic.operator_name
                    },
                    name = f"{formatted_time}_Cons_{self.args.use_consistancy}_{init}_pertub_{self.args.pertub}_task_{self.args.task}_lr_{self.args.learning_rate}_rank_{self.args.rank}_max_iter_unfold_{self.args.max_unfolded_iter}_lambda_sig{self.args.lambda_}",
                    
                )
            
        # resume training 
        self.resume_epoch = -1
        if self.args.resume_model:
            self._resume_model()

        # training loop
        for epoch in range(self.resume_epoch+1, epochs):
            # setting training mode
            self.logger.info(f"Epoch ==== >> {epoch}")
            self.model.train()
            for ii, batch_0 in enumerate(self.trainloader):
                # tracking monitor memory
                process = psutil.Process(os.getpid())
                self.logger.info(f"-----------------------------------------------------------")
                self.logger.info(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")
                self.logger.info(f"-----------------------------------------------------------")

                # start epoch run
                t_start = time.time()
                self.logger.info(f"Epoch ==== >> {epoch}\t|\t batch ==== >> {ii}\n") 
                with self.accelerator.accumulate(self.model):  
                    
                    
                    B,C = batch_0.shape[:2]
                    
                    # ---> Adversarial ground truths
                    valid = torch.Tensor(B, 1).fill_(1.0).to(self.device).requires_grad_(False)
                    fake = torch.Tensor(B, 1).fill_(0.0).to(self.device).requires_grad_(False)

                    # ----------------------------#
                    #  Finetuning Diffusion model #
                    # ----------------------------#
                    # ---> sample noisy x_t
                    t = self.diffusion_scheduler.sample_times(B, self.device)
                    noise_target = torch.randn_like(batch_0)
                    x_t = self.diffusion_scheduler.sample_xt(batch_0, t, noise=noise_target)
                    
                    # ---> observation y 
                    if self.args.task == "deblur":
                        blur = self.kernels.get_blur()
                        y = self.physic.y(batch_0, blur)
                    elif self.args.task=="sr":
                        blur = self.kernels.get_blur()
                        y = self.physic.y(batch_0, blur, self.args.physic.sr.sf)
                    elif self.task=="inp":
                        y = self.physic.y(batch_0)
                                       
                    # ---> initialisation unfolded model
                    if self.args.dpir.use_dpir and self.args.task == "deblur":
                        y_ = utils.inverse_image_transform(y)
                        x_init = self.dpir_model.run(y_, blur, iter_num = 1) #y.clone()
                        x_init = utils.image_transform(x_init)
                    else:
                        if self.args.task=="sr":
                            # utils.enable_disable_lora(self.model, enabled = False)
                            with torch.no_grad():
                                y_ = self.physic.upsample(utils.inverse_image_transform(y))
                                x_init = self.dpir_model.run(y_, blur, iter_num = 1) #y.clone()
                                x_init = utils.image_transform(y_)
                            # utils.enable_disable_lora(self.model, enabled = True)
                        else:
                            x_init = y.clone()
 
                    # ---> unfolded model
                    obs = {"x_t":x_t, "y":y}
                    if not self.args.task=="inp":
                        obs["blur"] = blur
                    out = self.hqs_module.run_unfolded_loop( obs, x_init, t, max_iter = self.max_unfolded_iter, lambda_sig = self.args.lambda_)
                    xstart_pred = out["xstart_pred"]
                    eps_pred = out["eps_pred"]
                    aux = out["aux"]
                    
                    # ---> save args
                    if epoch ==0 and ii == 0:
                        self.save_args_yaml()
                    
                    # ---> loss
                    loss = nn.MSELoss()(eps_pred, noise_target).mean() #nn.MSELoss()(xstart_pred, utils.inverse_image_transform(batch_0)).mean()
            
                    # ---> updating learnable parameters and setting gradient to zero
                    self.accelerator.backward(loss)        
                    self.optimizer.step() 
                    self.model.zero_grad()
                    
                    # ----------------------#
                    #  Train Discriminator  #
                    # ----------------------#
                    if self.args.adversarial:
                        self.optimizer_D.zero_grad()
                    
                        # ---> Loss for real images
                        real_pred = self.discriminator(utils.inverse_image_transform(batch_0))
                        d_real_loss = self.adversarial_loss(real_pred, valid)

                        # ---> Loss for fake images
                        fake_pred = self.discriminator(xstart_pred.detach())
                        d_fake_loss = self.adversarial_loss(fake_pred, fake)

                        # ---> Total discriminator loss
                        d_loss = (d_real_loss + d_fake_loss) / 2

                        d_loss.backward()
                        self.optimizer_D.step()
                    
                    # ---> update scheduler    
                    self.scheduler.step()


                with torch.no_grad():
                    loss_val = loss.clone().detach().cpu().mean().item()
                    self.logger.info(f"Loss: {loss_val}")            
                    
                    if self.args.use_wandb:
                        wandb.log({"loss": loss_val})
                        wandb.log(
                            {
                                "loss": loss_val,   
                                "mse": self.metrics.mse_function(
                                            xstart_pred.detach().cpu(), 
                                            utils.inverse_image_transform(batch_0).detach().cpu()
                                        ).item(), 
                                "psnr": self.metrics.psnr_function(
                                            xstart_pred.detach().cpu(), 
                                            utils.inverse_image_transform(batch_0).detach().cpu()
                                        ).item()
                            }
                        )
                        
                        # 
                        sigma_t = extract_tensor(self.diffusion_scheduler.sigmas, t, x_t.shape).ravel()[0]
                        sqrt_alphas_cumprod_t = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape).ravel()[0]
                        if ii%100 == 0:
                            if self.args.use_wandb:
                                xstart_pred_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(xstart_pred), 
                                        caption=f"x0_hat Epoch: {ii} sigma_t: {sigma_t}\t alpha: {sqrt_alphas_cumprod_t}\t 1 / sigima^2 alpha: {1 / (sigma_t**2*sqrt_alphas_cumprod_t)}\tt:{t.ravel()[0]}", 
                                        file_type="png"
                                    )
                                batch_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(batch_0), 
                                        caption=f"x0 Epoch: {ii} ", 
                                        file_type="png"
                                    )
                                ys_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(y), 
                                        caption=f"y Epoch: {ii}", 
                                        file_type="png"
                                    )
                                xt_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(x_t), 
                                        caption=f"xt Epoch: {ii}", 
                                        file_type="png"
                                    )
                                z_input_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(out["z_input"]), 
                                        caption=f"z_input", 
                                        file_type="png"
                                    )
                                aux_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(aux), 
                                        caption=f"z Epoch: {ii}", 
                                        file_type="png"
                                    )

                                wandb.log(
                                    {
                                        "ref image": batch_wdb, 
                                        "blurred":ys_wdb, 
                                        "xt":xt_wdb, 
                                        "x predict": xstart_pred_wdb, 
                                        "Auxiliary":aux_wdb,
                                        "z_input_wdb":z_input_wdb,
                                    }
                                )
                    end_time = time.time()  
                    ellapsed = end_time - t_start       
                    self.logger.info(f"Elapsed time per epoch: {ellapsed}") 
            
            # reporting some images 
            if epoch%1 == 0:
                if self.args.use_wandb:
                    xstart_pred_wdb = wandb.Image(
                            utils.get_batch_rgb_from_batch_tensor(xstart_pred), 
                            caption=f"x0_hat Epoch: {epoch}", 
                            file_type="png"
                        )
                    batch_wdb = wandb.Image(
                            utils.get_batch_rgb_from_batch_tensor(batch_0), 
                            caption=f"x0 Epoch: {epoch}", 
                            file_type="png"
                        )
                    ys_wdb = wandb.Image(
                            utils.get_batch_rgb_from_batch_tensor(y), 
                            caption=f"y Epoch: {epoch}", 
                            file_type="png"
                        )
                    xt_wdb = wandb.Image(
                            utils.get_batch_rgb_from_batch_tensor(x_t), 
                            caption=f"xt Epoch: {epoch}", 
                            file_type="png"
                        )
                    aux_wdb = wandb.Image(
                            utils.get_batch_rgb_from_batch_tensor(aux), 
                            caption=f"z Epoch: {epoch}", 
                            file_type="png"
                        )

                    wandb.log(
                        {
                            "ref image": batch_wdb, 
                            "blurred":ys_wdb, 
                            "xt":xt_wdb, 
                            "x predict": xstart_pred_wdb, 
                            "Auxiliary":aux_wdb,
                        }
                    )
            torch.cuda.empty_cache() 
            
            # save checkpoints
            if epoch%self.args.save_checkpoint_range == 0:
                self.save_trainable_params(epoch)   
    
    def save_args_yaml(self):
        yaml_dir = bf.join(self.args.path_save, self.args.task, "Config", self.args.date)
        os.makedirs(yaml_dir, exist_ok=True)
        filename = f"lora_model_{self.args.task}_{self.args.physic.operator_name}.yaml"
        filepath = bf.join(yaml_dir, filename)
        with open(filepath, 'w') as f:
            yaml.dump(self.args, f)        
    #
    def eval_epoch(self):
        with torch.no_grad():
            self.model.eval()
            im = 0
            for ii, batch_val in enumerate(self.testloader):
                
                seq, progress_seq = self.diffusion_scheduler.get_seq_progress_seq(100)
                
                # observation y
                self.physic.step()
                y = self.physic.y(batch_val)
                x = torch.randn_like(batch_val)
                progress_img = []
                
                for ii in range(len(seq)):
                    tii = seq[ii]
                    # unfolding: xstrat_pred, eps_pred, auxiliary variable
                    sigmas_tii = extract_tensor(self.diffusion_scheduler.sigmas, tii, y.shape)
                    x_init = (
                        sigmas_tii / (sigmas_tii + self.physic.sigma_model) * x + 
                        self.physic.sigma_model / (sigmas_tii + self.physic.sigma_model) * y
                    )
                    out_val = self.hqs_module.run_unfolded_loop( y, x_init, x, tii, max_iter = self.max_unfolded_iter)
                    x0 = out_val["xstart_pred"]
                    eps = out_val["eps_pred"]
                    z0 = out_val["aux"]
                    
                    # sample from the predicted noise
                    if seq[ii] != seq[-1]: 
                        beta_t = self.diffusion_scheduler.betas[seq[ii]]    
                        sqrt_1m_alphas_cumprodtii = self.diffusion_scheduler.sqrt_1m_alphas_cumprod[seq[ii]]    
                        sqrt_1m_alphas_cumprodtiim1 = self.diffusion_scheduler.sqrt_1m_alphas_cumprod[seq[ii+1]]    
                        eta_sigma = sqrt_1m_alphas_cumprodtiim1 / sqrt_1m_alphas_cumprodtii * torch.sqrt(beta_t)       
                        x = (1 / math.sqrt(1 - beta_t)) * ( x - beta_t * eps / sqrt_1m_alphas_cumprodtii) + eta_sigma.sqrt() * torch.randn_like(x)
                    else:
                        x = x0
        
                    # transforming pixels from [-1,1] ---> [0,1]    
                    x_0 = utils.inverse_image_transform(x)
                    
                    # save the process
                    if self.args.save_progressive and (seq[ii] in progress_seq):
                        x_show = utils.get_rgb_from_tensor(x_0)      #[0,1]
                        progress_img.append(x_show)
                
                if self.args.save_progressive:   
                    img_total = cv2.hconcat(progress_img)
                    if self.args.use_wandb:
                        image_wdb = wandb.Image(img_total, caption=f"progress sequence for im {im}", file_type="png")
                        wandb.log({"pregressive":image_wdb})
                    
                if self.args.use_wandb:
                    x0_wdb = wandb.Image(
                        utils.get_rgb_from_tensor(x_0), 
                        caption=f"pred x start {im}", 
                        file_type="png"
                    )
                    z0_wdb = wandb.Image(
                        utils.get_rgb_from_tensor(z0), 
                        caption=f"latent variable {im}", 
                        file_type="png"
                    )
                    
                    y_wdb = wandb.Image(
                        utils.get_rgb_from_tensor(y), 
                        caption=f"observation {im}", 
                        file_type="png"
                    )
                    xy = [y_wdb, x0_wdb, z0_wdb]
                    wandb.log({"blurred and predict":xy})
        
    # - save the checkpoint model   
    def save(self, epoch):
        checkpoint_dir = bf.join(self.args.path_save, self.args.task, "Checkpoints", self.args.date, f"model_noise_{self.args.physic.sigma_model}")
        try:
            unwrap_model = self.accelerator.unwrap_model(self.model)
        except:
            unwrap_model = self.model
            
        if self.accelerator.is_main_process:
            self.logger.info(f"Saving checkpoint {epoch}...")
            filename = f"{self.args.task}_{self.args.physic.operator_name}_model_{(epoch):03d}.pt"
            
            filepath = bf.join(checkpoint_dir, filename)
            with bf.BlobFile(filepath, "wb") as f:
                torch.save({
                        'model_state_dict': unwrap_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                    }, 
                f
            )

        del unwrap_model
        
    # saving only trainable parameters
    def save_trainable_params(self, epoch):
        checkpoint_dir = bf.join(self.args.path_save, self.args.task, "Checkpoints", self.args.date, f"model_noise_{self.args.physic.sigma_model}") 
        try:
            unwrap_model = self.accelerator.unwrap_model(self.model)
            unwrap_discriminator = self.accelerator.unwrap_model(self.discriminator)
        except:
            unwrap_model = self.model
            unwrap_discriminator = self.discriminator
        
        # state dict 
        trainable_lora_state_dict = {name: param for name, param in unwrap_model.named_parameters() if param.requires_grad}  
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Saving checkpoint {epoch}...")
            filename = f"LoRA_model_{self.args.task}_{self.args.physic.operator_name}_{(epoch):03d}.pt"
            if self.args.task=="sr":
                filename = f"LoRA_model_{self.args.task}_factor_{self.args.physic.sr.sf}_{(epoch):03d}.pt"
                
            
            filepath = bf.join(checkpoint_dir, filename)
            with bf.BlobFile(filepath, "wb") as f:
                torch.save(
                    {
                        'model_state_dict': trainable_lora_state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),  
                        'discriminator_state_dict': self.discriminator.state_dict(),  
                        "epoch":epoch
                    }, 
                    f
                )
            
    
    def load_trainable_params(self, epoch):
        checkpoint_dir = bf.join(self.args.path_save, self.args.task, "Checkpoints", self.args.date, f"model_noise_{self.args.physic.sigma_model}")
        if self.task=="sr":
            filename = f"LoRA_model_{self.args.task}_{self.args.physic.sr.sf}_{(epoch):03d}.pt"
        else:  
            filename = f"LoRA_model_{self.args.task}_{self.args.physic.operator_name}_{(epoch):03d}.pt"
        filepath = bf.join(checkpoint_dir, filename)
        trainable_state_dict = torch.load(filepath)
        self.copy_model.load_state_dict(trainable_state_dict["model_state_dict"], strict=False)
        
        if trainable_state_dict["model_state_dict"] is not None:
            if hasattr(self, "discriminator"):
                self.discriminator.load_state_dict(trainable_state_dict["discriminator_state_dict"], strict=False)
                
        
    # resume training 
    def _resume_model(self, resume_date = None, resume_epoch = None):
        dir_ = bf.join(self.args.path_save, self.args.task, "Checkpoints")
        
        if self.args.resume_model:
            if resume_date is not None:
                checkpoint_dir = bf.join(dir_, resume_date)
            else:
                resume_dates = os.listdir(dir_)
                if len(resume_dates)>0:
                    resume_date = sorted(resume_dates)[-1]
                    checkpoint_dir = bf.join(dir_, resume_date)
                else:
                    self.resume_epoch = 0
                    self.logger.info(f"This folder is empty: {resume_date}. We will restart the training!!")
                    return -1

            self.logger.info(f"Resume date ====> {resume_date}")
            self.resume_date = resume_date
            # Filter files that start with 'LoRA_model' and end with '.pt'
            filtered_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("LoRA_model_debur") and f.endswith(".pt")]
            if len(filtered_files)>0:
                if resume_epoch is None:
                    # recent checkpoint base on epoch
                    filename = sorted(filtered_files)[-1]
                    file_path = bf.join(checkpoint_dir, filename)
                    resume_epoch = np.int32(filename.rsplit("_",1)[-1].split(".")[0])
                else:
                    filename = f"LoRA_model_{self.args.task}_{(resume_epoch):03d}.pt"
                    file_path = bf.join(checkpoint_dir, filename)
                self.logger.info(f"Resume epoch ====>> {resume_epoch}")
                    
                self.resume_epoch=resume_epoch
                if os.path.exists(file_path):
                    checkpoint = torch.load(file_path)
                    self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    self.resume_epoch = resume_epoch#checkpoint["epoch"]
                    self.logger.info(f"We are resuming training from epoch: {self.resume_epoch}.")
                    print(f"We are resuming training from epoch: {self.resume_epoch}.")
                else:
                    self.logger.info(f"This file ({filename}) do not exist. We will start training from epoch 0 !!!")
                    self.resume_epoch=0
                    return -1
            else:
                self.logger.info(f"This folder ({resume_date}) do not contains any checkpoints. We will restart the training  from epoch 0 !!!")
                self.resume_epoch=0
                return -1        
    
    def _seed(self, seed):
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # For GPU operations, set the seed for CUDA as well
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you're using multi-GPU

        # Ensure that deterministic algorithms are used for certain operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False          
               
# ---        
def model_parameters(model, logger):
    trained_params = 0
    Total_params = 0
    frozon_params = 0
    for p in model.parameters():
        Total_params += p.numel()
        if p.requires_grad:
            trained_params += p.numel()
        else:
            frozon_params += p.numel()
            
    logger.info(f"Total parameters ====> {Total_params}")
    logger.info(f"Total frozen parameters ====> {frozon_params}")
    logger.info(f"Total trainable parameters ====> {trained_params}")

