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
        physic:Union[phy.Deblurring, phy.Inpainting],
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
        self.max_unfolded_iter = max_unfolded_iter
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.hqs_module = hqs_module
        
        self.copy_model = copy.deepcopy(self.model)
        
        # set seed
        self._seed(self.args.seed)
        
        # Accelerator: DDP
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        
        # set training
        self._set_training()
        self.metrics = utils.Metrics(self.device)
        
    def _set_training(self):
        model_parameters(self.model, self.logger)
        self.model_params_learnable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = AdamW(self.model_params_learnable, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(100 * 0.2), gamma=0.1)
        self.model, self.optimizer, self.trainloader, self.testloader, self.scheduler = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.trainloader,
            self.testloader, 
            self.scheduler, 
        )
        
        # HQS model
        self.denoising_timestep = GetDenoisingTimestep(self.device)
        self.hqs_model = HQS_models(
            self.model,
            self.physic, 
            self.diffusion_scheduler, 
            self.denoising_timestep,
            self.args
        )
        
        # DPIR model
        self.dpir_model = DPIR_deb(device = self.device, model_name = self.args.dpir.model_name, pretrained_pth=self.args.dpir.pretrained_pth)
    
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
            wandb.init(
                    # set the wandb project where this run will be logged
                    project=f"Training Unfolded Conditional Diffusion Models {self.args.task}",
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
                        "random blur":self.args.physic.random_blur,
                        "blur name": self.args.physic.blur_name if not self.args.physic.random_blur else None
                    },
                    name = f"{formatted_time}_Cons_{self.args.use_consistancy}_{init}_pertub_{self.args.pertub}_task_{self.args.task}_lr_{self.args.learning_rate}_rank_{self.args.rank}_max_iter_unfold_{self.args.max_unfolded_iter}",
                    
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
                    # sample noisy x_t
                    B = batch_0.shape[0]
                    t = self.diffusion_scheduler.sample_times(B, self.device)
                    noise_target = torch.randn_like(batch_0)
                    x_t = self.diffusion_scheduler.sample_xt(batch_0, t, noise=noise_target)
                    
                    # observation y 
                    y = self.physic.y(batch_0)
                    
                    # unfolding: xstrat_pred, eps_pred, auxiliary variable
                    sigmas_t = extract_tensor(self.diffusion_scheduler.sigmas, t, y.shape)
                   
                    # initialisation unfolded model
                    if self.args.init_xt_y:
                        x_init = (
                            sigmas_t**2 / (sigmas_t**2 + self.physic.sigma_model**2) * x_t + 
                            self.physic.sigma_model**2 / (sigmas_t**2 + self.physic.sigma_model**2) * y
                        )
                    else:
                        x_init = y.clone()
                    if self.args.dpir.use_dpir:
                        y_ = utils.inverse_image_transform(y)
                        x_init = self.dpir_model.run(y_, self.physic.blur, iter_num = 1) #y.clone()
                        x_init = utils.image_transform(x_init)
                    
                    # unfolded model
                    out = self.hqs_module.run_unfolded_loop( y, x_init, x_t, t, max_iter = self.max_unfolded_iter, lambda_ = self.args.lambda_)
                    xstart_pred = out["xstart_pred"]
                    eps_pred = out["eps_pred"]
                    aux = out["aux"]
                    
                    # save args
                    if epoch ==0 and ii == 0:
                        self.save_args_yaml()
                    
                    # loss
                    loss = nn.MSELoss()(eps_pred, noise_target).mean()
            
                    # updating learnable parameters and setting gradient to zero
                    self.accelerator.backward(loss)        
                    self.optimizer.step()
                    self.scheduler.step() 
                    self.model.zero_grad()
            
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
                        if ii%100 == 0:
                            if self.args.use_wandb:
                                xstart_pred_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(xstart_pred), 
                                        caption=f"x0_hat Epoch: {ii}", 
                                        file_type="png"
                                    )
                                batch_wdb = wandb.Image(
                                        utils.get_batch_rgb_from_batch_tensor(batch_0), 
                                        caption=f"x0 Epoch: {ii}", 
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
        filename = f"lora_model_{self.args.task}.yaml"
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
            filename = f"{self.args.task}_model_{(epoch):03d}.pt"
            
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
        except:
            unwrap_model = self.model
        
        # state dict 
        trainable_lora_state_dict = {name: param for name, param in unwrap_model.named_parameters() if param.requires_grad}  
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Saving checkpoint {epoch}...")
            filename = f"LoRA_model_{self.args.task}_{(epoch):03d}.pt"
            
            filepath = bf.join(checkpoint_dir, filename)
            with bf.BlobFile(filepath, "wb") as f:
                torch.save({
                        'model_state_dict': trainable_lora_state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),  
                        "epoch":epoch,
                    }, 
                f
            )
    
    def load_trainable_params(self, epoch):
        checkpoint_dir = bf.join(self.args.path_save, self.args.task, "Checkpoints", self.args.date, f"model_noise_{self.args.physic.sigma_model}")
        filename = f"LoRA_model_{self.args.task}_{(epoch):03d}.pt"
        filepath = bf.join(checkpoint_dir, filename)
        trainable_state_dict = torch.load(filepath)
        self.copy_model.load_state_dict(trainable_state_dict["model_state_dict"], strict=False)
        
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

