import math
import numpy as np
import random
import copy
import os
import psutil
import wandb
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import blobfile as bf
import datetime
import kornia
import copy
import yaml

from torchvision.utils import save_image
from accelerate import Accelerator
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .unrolledPhysics import UniformMotionBlurGenerator
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

# diffusion scheduler module        
# class DiffusionScheduler:
#     def __init__(
#         self, 
#         start_beta:float = 0.1*1e-3, 
#         end_beta: float = 20*1e-3, 
#         num_timesteps:int = 1000,
#         dtype = torch.float32,
#         device = "cpu", 
#         fix_timestep = True,
#         noise_schedule_type:str = "linear"
#     ) -> None:
#         self.fix_timestep = fix_timestep
#         self.num_timesteps = num_timesteps
#         if noise_schedule_type == "cosine":
#             betas = CosineScheduler(num_timesteps)
#         elif noise_schedule_type=="linear":
#             betas = LinearScheduler(num_timesteps, beta_start=start_beta, beta_end=end_beta)
#         else:
#             raise NotImplementedError("This noise schedule is not implemented !!!")
        
#         self.betas = torch.from_numpy(betas).to(dtype).to(device)
#         self.alphas                  = 1.0 - self.betas
#         self.alphas_cumprod          = torch.from_numpy(np.cumprod(self.alphas.cpu().numpy(), axis=0)).to(device)
#         self.sqrt_alphas_cumprod     = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_1m_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod)
#         self.sigmas   = torch.div(self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod)
        
#     def sample_times(self, bzs, device):
#         if self.fix_timestep:
#             t_ = torch.randint(1, self.num_timesteps, (1,))
#             t = (t_ * torch.ones((bzs,))).long().to(device)
#         else:
#             t = torch.randint(
#                 1, 
#                 self.num_timesteps, 
#                 (bzs,), 
#                 device=device
#             ).long()
        
#         return t
    
#     def sample_xt(
#         self, 
#         x_start:torch.Tensor, 
#         timesteps:torch.Tensor, 
#         noise:torch.Tensor
#     ) -> torch.Tensor:
        
#         if noise is None:
#             noise = torch.randn_like(x_start)
#         assert noise.shape == x_start.shape
#         return (
#             extract_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
#             + extract_tensor(self.sqrt_1m_alphas_cumprod, timesteps, x_start.shape)
#             * noise
#         )   
    
#     #predict x_start from eps 
#     def predict_xstart_from_eps(self, x_t, t, eps):
#         assert x_t.shape == eps.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {eps.shape})")
#         return (
#             (x_t - extract_tensor( self.sqrt_1m_alphas_cumprod, t, x_t.shape ) * eps) / 
#             extract_tensor(self.sqrt_alphas_cumprod, t, x_t.shape )
#         )
        
#     # predict eps from x_start
#     def predict_eps_from_xstrat(self, x_t, t, xstart):
#         assert x_t.shape == xstart.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {xstart.shape})")
#         return (
#             (x_t - extract_tensor(self.sqrt_alphas_cumprod, t, x_t.shape ) * xstart) / 
#             extract_tensor( self.sqrt_1m_alphas_cumprod, t, x_t.shape )
            
#         )
    
#     # get sequence of timesteps 
#     def get_seq_progress_seq(self, iter_num):
#         seq = np.sqrt(np.linspace(0, self.num_timesteps**2, iter_num))
#         seq = [int(s) for s in list(seq)]
#         seq[-1] = seq[-1] - 1
#         progress_seq = seq[::max(len(seq)//10,1)]
#         if progress_seq[-1] != seq[-1]:
#             progress_seq.append(seq[-1])
#         seq = seq[::-1]
#         progress_seq = progress_seq[::-1]
#         return (seq, progress_seq)

# # coustomise timesteps for the denoising step  
# class GetDenoisingTimestep:
#     def __init__(self, device, noise_schedule_type = "linear"):
#         self.scheduler = DiffusionScheduler(device=device,noise_schedule_type = noise_schedule_type)
#         self.device = device
    
#     # getting differently the timestep   
#     def get_tz_ty_rho(self, t, sigma_y, x_shape):
#         print(f"sigma y {sigma_y}\n\n")
#         sigma_t = self.scheduler.sigmas.to(self.device)[t]
#         if not torch.is_tensor(sigma_y):
#             sigma_y = torch.tensor(sigma_y).to(self.device)
#         else:
#             sigma_y = sigma_y.to(self.device)
#         ty = self.t_y(sigma_y)
#         a1 = sigma_y / (sigma_y + sigma_t)
#         a2 = sigma_t / (sigma_y + sigma_t)
#         tz = (ty * a1 + t * a2).long()
#         rho = extract_tensor(self.scheduler.sigmas, tz, x_shape)
#         return rho, tz, ty
    
#     # get rho and t from sigma_t and sigma_y
#     def get_tz_rho(self, t, sigma_y, x_shape, T_AUG):
#         sigma_t = extract_tensor(self.scheduler.sigmas, t, x_shape) 
#         if not torch.is_tensor(sigma_y):
#             sigma_y = torch.tensor(sigma_y).to(self.device)
#         else:
#             sigma_y = sigma_y.to(self.device)
#         rho = sigma_y * sigma_t * torch.sqrt(1 / (sigma_y**2 + sigma_t**2))
#         tz = torch.clamp(
#             self.t_y(rho[0,0,0,0]).cpu() - torch.tensor(T_AUG),
#             1,
#             self.scheduler.num_timesteps
#         )
#         tz = (tz * torch.ones((x_shape[0],))).long().to(self.device)
#         rho_ = extract_tensor(self.scheduler.sigmas, tz, x_shape)
#         ty = self.t_y(sigma_y)
#         return rho_, tz, ty
    
#     # getting the corresponding timestep from the diffusion process   
#     def t_y(self, sigma_y):
#         t_y = torch.clamp(torch.abs(self.scheduler.sigmas.to(self.device) - sigma_y).argmin(), 1, self.scheduler.num_timesteps)
#         return t_y    
    
# physic module        
class physics_models:
    def __init__(
        self, 
        kernel_size:int, 
        blur_name:str, 
        random_blur: bool = False,
        device:str = "cpu",
        sigma_model:float= 0.05, 
        dtype = torch.float32, 
        transform_y:bool = False,
        path_motion_kernel:str = "/users/cmk2000/sharedscratch/Datasets/blur-kernels/kernels",
        mode = "test",
    ) -> None:
        # parameters
        self.transform_y = transform_y
        self.device = device
        self.sigma_model = 2 * sigma_model if self.transform_y else sigma_model
        self.dtype = dtype
        self.kernel_size = kernel_size
        self.blur_name = blur_name
        self.random_blur = random_blur
        self.mode = mode
        self.uniform_kernel_generator = UniformMotionBlurGenerator(kernel_size=(kernel_size,kernel_size))
        
        # motion kernel
        if mode == "train":
            self.path_motion_kernel_train = os.path.join(path_motion_kernel, "training_set")
            self.list_motion_kernel = os.listdir(self.path_motion_kernel_train)
        else:
            self.path_motion_kernel_val = os.path.join(path_motion_kernel, "validation_set")
            self.list_motion_kernel = os.listdir(self.path_motion_kernel_val)
        
        # defined transforms 
        self.transforms_kernel = transforms.Compose([
                transforms.Resize((self.kernel_size, self.kernel_size)),
                transforms.ToTensor()
            ])
        
    def _get_kernel(self):
        if self.blur_name == "gaussian":
            blur_kernel = self._gaussian_kernel(self.kernel_size).to(self.device)
        elif self.blur_name  == "uniform":
            blur_kernel = self._uniform_kernel(self.kernel_size).to(self.device)
        elif self.blur_name  == "motion":
            blur_kernel = self._motion_kernel(self.kernel_size).to(self.device)
        elif self.blur_name == "uniform_motion":
            blur_kernel = self._uniform_motion_kernel().to(self.device)
        else:
            raise ValueError(f"Blur type {self.blur_name } not implemented !!")
        return blur_kernel

    def _uniform_motion_kernel(self):
        return self.uniform_kernel_generator.step(1)["filter"]

    def _motion_kernel(self):
        if self.mode == "train":
            kernel_name = random.choice(self.list_motion_kernel) 
            kernel = Image.open(os.path.join(self.path_motion_kernel_train, kernel_name)) 
        else:
            kernel_name = self.list_motion_kernel[7]  
            kernel = Image.open(os.path.join(self.path_motion_kernel_val, kernel_name))
            
        kernel_ = self.transforms_kernel(kernel)[0,...].squeeze()
        kernel_ /= kernel_.sum()
        return kernel_
    
    def _gaussian_kernel(self, kernel_size, std = 3.0):
        c = int((kernel_size - 1)/2)
        v = torch.arange(-kernel_size + c, kernel_size - c, dtype = self.dtype).reshape(-1,1)
        vT = v.T
        exp = torch.exp(-0.5*(v**2 + vT**2)/std)
        exp /= torch.sum(exp)
        return exp 
        
    def _uniform_kernel(self, kernel_size):
        exp = torch.ones(kernel_size, kernel_size).to(self.dtype)
        exp /= torch.sum(exp)
        return exp
    
    def blur2A(self, im_size):
        if not hasattr(self, "blur"):
            self.blur = self._get_kernel()
        else:
            pass
        
        if len(self.blur.shape) > 2:
            raise ValueError("The kernel dimension is not correct")
        device = self.blur.device
        n, m = self.blur.shape
        H_op = torch.zeros(im_size, im_size, dtype=self.blur.dtype).to(device)
        H_op[:n,:m] = self.blur
        for axis, axis_size in enumerate(self.blur.shape):
            H_op = torch.roll(H_op, -int(axis_size / 2), dims=axis)
        FFT_h = torch.fft.fft2(H_op, dim = (-2,-1))
        return FFT_h[None, None, ...], self.blur

    # forward operator
    def Ax(self, x):
        im_size = x.shape[-1]
        FFT_h, h = self.blur2A(im_size)
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        ax = torch.real(
            torch.fft.ifft2(FFT_h * FFT_x, dim = (-2,-1))
        )
        return ax
    
    # forward operator
    def ATx(self, x):
        im_size = x.shape[-1]
        FFT_h, h = self.blur2A(im_size)
        FFT_hc = torch.conj(FFT_h)
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        atx = torch.real(
            torch.fft.ifft2(FFT_hc * FFT_x, dim = (-2,-1))
        )
        return atx
    
    # observation y
    def y(self, x:torch.Tensor)->torch.Tensor:
        if not self.transform_y:
            x0 = utils.inverse_image_transform(x)
        else:
            x0 = x
        self.blur = self._get_kernel()
        out = self.Ax(x0) + self.sigma_model * torch.randn_like(x0)
        return out

# Half-Quadratic Splitting module       
class HQS_modelss:
    def __init__(
        self,
        model, 
        physic:Union[phy.Deblurring, phy.Inpainting], 
        diffusion_scheduler:DiffusionScheduler, 
        get_denoising_timestep:GetDenoisingTimestep,
        args
    ) -> None:
        
        self.model = model
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
        self.get_denoising_timestep = get_denoising_timestep
        self.args = args
    
    def proxf_update(
        self, 
        y:torch.Tensor,
        x:torch.Tensor, 
        x_t:torch.Tensor, 
        rho:float, 
        t:int,
        lambda_:float,
    )-> torch.Tensor:
        
        sigma_t = extract_tensor(self.diffusion_scheduler.sigmas, t, x_t.shape)
        sqrt_alpha_comprod = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape)
    
        # self.im_size = y.shape[-1]
        # Aty = self.physic.ATx(y)
        # FFTAty = torch.fft.fft2(Aty, dim = (-2,-1))        
        # FFT_h,_ = self.physic.blur2A(self.im_size)
        # FFTx = torch.fft.fft2(x, dim=(-2,-1))
        # FFTx_t = torch.fft.fft2(x_t, dim=(-2,-1))
        # _mu = FFTAty / (lambda_*self.physic.sigma_model**2) + FFTx / rho**2 + FFTx_t  / (sigma_t**2 * sqrt_alpha_comprod)
        # estimate_temp = torch.real(torch.fft.ifft2(_mu * self._precision(FFT_h, rho, sigma_t, lambda_),dim = (-2,-1)))
        estimate_temp = self.physic.mean_estimate(self, x_t, y, x, self.physic.sigma_model, sigma_t, sqrt_alpha_comprod, rho,lambda_)
        return estimate_temp
        
    # def _precision(
    #     self,
    #     FFT_h:torch.Tensor, 
    #     rho:float, 
    #     sigma_t:float,
    #     lambda_:float
    #     ) -> torch.Tensor:
        
    #     out =  (
    #         FFT_h**2 / (lambda_*self.physic.sigma_model**2) 
    #         + 1 / sigma_t**2
    #         + 1 / rho**2 
    #     )
    #     return 1 / out
    
    def run_unfolded_loop(
        self, 
        y:torch.Tensor, 
        x:torch.Tensor, 
        x_t:torch.Tensor, 
        t:int, 
        max_iter:int = 3,
        T_AUG:int = 0,
        lambda_:float = 1.0
    ) -> dict:
        B,C = x.shape[:2]
        
        # get denoising timestep
        rho, t_denoising, ty = self.get_denoising_timestep.get_tz_rho(t, self.physic.sigma_model, x_t.shape, T_AUG)
        
        # --- initialisation
        x = x / extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t_denoising, x.shape)
            
        if self.args.use_wandb and self.args.mode == "train":
            wandb.log(
                {
                    "tz":t_denoising.squeeze().max().cpu().numpy(), 
                    "rho":rho.squeeze().min().cpu().numpy(),
                }
            )
            
        for _ in range(max_iter):
            # prox step
            z_ = self.proxf_update(y, x, x_t, rho, t, lambda_)

            # noise prediction
            if self.args.pertub:
                noise = torch.randn_like(z_)
                val = 1 * torch.ones_like(t_denoising).long()
                max_val = 1000 * torch.ones_like(t_denoising).long()
                tu = torch.clamp(t_denoising, val, max_val)
                z = self.diffusion_scheduler.sample_xt(z_, tu, noise=noise)
            else:
                z = z_.clone()
                
            eps = self.model(z, t_denoising)
            eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_t.shape[2:]) else (eps, eps)
            
            # estimate of x_start
            x = self.diffusion_scheduler.predict_xstart_from_eps(z, t_denoising, eps)
            x_pred = utils.inverse_image_transform(x)
            x_ = torch.clamp(x_pred, 0.0, 1.0)
            
        pred_eps = self.diffusion_scheduler.predict_eps_from_xstrat(x_t, t, x)
        return {
            "xstart_pred":x_, 
            "aux":utils.inverse_image_transform(z_), 
            "eps_pred":pred_eps,
            "eps_z":eps
        }

# trainer model
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
                    project="Training Unfolded Conditional Diffusion Models",
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
                    out = self.hqs_module.run_unfolded_loop( y, x_init, x_t, t, max_iter = self.max_unfolded_iter)
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
   
                    end_time = time.time()  
                    ellapsed = end_time - t_start       
                    self.logger.info(f"Elapsed time per epoch: {ellapsed}") 
            
            # reporting some images 
            if epoch%5 == 0:
                if self.args.use_wandb:
                    xstart_pred_wdb = wandb.Image(
                            get_batch_rgb_from_batch_tensor(xstart_pred), 
                            caption=f"x0_hat Epoch: {epoch}", 
                            file_type="png"
                        )
                    batch_wdb = wandb.Image(
                            get_batch_rgb_from_batch_tensor(batch_0), 
                            caption=f"x0 Epoch: {epoch}", 
                            file_type="png"
                        )
                    ys_wdb = wandb.Image(
                            get_batch_rgb_from_batch_tensor(y), 
                            caption=f"y Epoch: {epoch}", 
                            file_type="png"
                        )
                    xt_wdb = wandb.Image(
                            get_batch_rgb_from_batch_tensor(x_t), 
                            caption=f"xt Epoch: {epoch}", 
                            file_type="png"
                        )
                    aux_wdb = wandb.Image(
                            get_batch_rgb_from_batch_tensor(aux), 
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
                        x_show = get_rgb_from_tensor(x_0)      #[0,1]
                        progress_img.append(x_show)
                
                if self.args.save_progressive:   
                    img_total = cv2.hconcat(progress_img)
                    if self.args.use_wandb:
                        image_wdb = wandb.Image(img_total, caption=f"progress sequence for im {im}", file_type="png")
                        wandb.log({"pregressive":image_wdb})
                    
                if self.args.use_wandb:
                    x0_wdb = wandb.Image(
                        get_rgb_from_tensor(x_0), 
                        caption=f"pred x start {im}", 
                        file_type="png"
                    )
                    z0_wdb = wandb.Image(
                        get_rgb_from_tensor(z0), 
                        caption=f"latent variable {im}", 
                        file_type="png"
                    )
                    
                    y_wdb = wandb.Image(
                        get_rgb_from_tensor(y), 
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
def get_rgb_from_tensor(x):
    x0_np = x.detach().cpu().numpy()       #[0,1]
    x0_np = np.squeeze(x0_np)
    if x0_np.ndim == 3:
        x0_np = np.transpose(x0_np, (1, 2, 0))
    return x0_np

# ---
def get_batch_rgb_from_batch_tensor(x):
    L = [get_rgb_from_tensor(xi) for xi in x]
    out= cv2.hconcat(L)*255.
    return out
      
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

