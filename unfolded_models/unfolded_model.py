import math
import numpy as np
import copy
import os
import psutil
import wandb
import time
import cv2
import matplotlib.pyplot as plt
import blobfile as bf

from torchvision.utils import save_image
from accelerate import Accelerator
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import Metrics, inverse_image_transform, image_transform
from utils_lora.lora_parametrisation import LoRa_model, enable_disable_lora

# extract into tensor       
def extract_tensor(
        arr:torch.Tensor, 
        timesteps:torch.Tensor, 
        broadcast_shape:torch.Size
    ) -> torch.Tensor:
        out = arr.to(device=timesteps.device)[timesteps].float()
        while len(out.shape) < len(broadcast_shape):
            out = out[..., None]
        return out.expand(broadcast_shape)

# diffusion scheduler module        
class DiffusionScheduler:
    def __init__(
        self, 
        start_beta:float = 0.1*1e-3, 
        end_beta: float = 20*1e-3, 
        num_timesteps:int = 1000,
        dtype = torch.float32,
        device = "cpu"
    ) -> None:
        self.num_timesteps = num_timesteps
        betas = np.linspace(start_beta, end_beta, num_timesteps)
        self.betas = torch.from_numpy(betas).to(dtype)
        self.alphas                  = 1.0 - self.betas
        self.alphas_cumprod          = torch.from_numpy(np.cumprod(self.alphas.cpu().numpy(), axis=0)).to(device)
        self.sqrt_alphas_cumprod     = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod)
        self.sigmas   = torch.div(self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod)
        
    def sample_times(self, bzs, device):
        t = torch.randint(
            0, 
            self.num_timesteps, 
            (bzs,), 
            device=device
        )
        t = t.long()
        return t
    
    def sample_xt(
        self, 
        x_start:torch.Tensor, 
        timesteps:torch.Tensor, 
        noise:torch.Tensor
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract_tensor(self.sqrt_1m_alphas_cumprod, timesteps, x_start.shape)
            * noise
        )    
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape, ValueError(f"x_t and eps have different shape ({x_t.shape} != {eps.shape})")
        return (
            (x_t- extract_tensor( self.sqrt_1m_alphas_cumprod, t, x_t.shape ) * eps) / 
            extract_tensor(self.sqrt_alphas_cumprod, t, x_t.shape )
        )

    def get_seq_progress_seq(self, iter_num):
        seq = np.sqrt(np.linspace(0, self.num_timesteps**2, iter_num))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1
        progress_seq = seq[::max(len(seq)//10,1)]
        if progress_seq[-1] != seq[-1]:
            progress_seq.append(seq[-1])
        seq = seq[::-1]
        progress_seq = progress_seq[::-1]
        return (seq, progress_seq)
# physic module        
class physics_models:
    def __init__(
        self, 
        kernel_size:int, 
        blur_name:str, 
        device = "cpu",
        sigma_model:float= 0.05, 
        dtype = torch.float32
    ) -> None:
        self.device = device
        self.sigma_model = sigma_model
        self.dtype = dtype
        self.kernel_size = kernel_size
        if blur_name == "gaussian":
            self.blur_kernel = self._gaussian_kernel(kernel_size).to(self.device)
        elif blur_name == "motion":
            self.blur_kernel = self._motion_kernel(kernel_size).to(self.device)
        else:
            raise ValueError(f"Blur type {blur_name} not implemented !!")
        
    def _gaussian_kernel(self, kernel_size, std = 3.0):
        c = int((kernel_size - 1)/2)
        x = torch.arange(-kernel_size + c, kernel_size - c, dtype = self.dtype).reshape(-1,1)
        xT = x.T
        exp = torch.exp(-0.5*(x**2+xT**2)/std)
        exp /= torch.sum(exp)
        return exp
        
    def _motion_kernel(self, kernel_size):
        exp = torch.ones(kernel_size, kernel_size).to(self.dtype)
        exp /= torch.sum(exp)
        return exp
    
    def blur2A(self, im_size):
        blur = self.blur_kernel.squeeze()
        if len(blur.shape) > 2:
            raise ValueError("The kernel dimension is not correct")
        device = blur.device
        n, m = blur.shape
        H = torch.zeros(im_size, im_size, dtype=blur.dtype).to(device)
        H[:n,:m] = blur
        for axis, axis_size in enumerate(blur.shape):
            H = torch.roll(H, -int((axis_size) / 2), dims=axis)
        FFT_H = torch.fft.fft2(H, dim = (-2,-1))
        return FFT_H[None, None, ...]
    
    def Ax(self, x):
        im_size = x.shape[-1]
        FFT_H = self.blur2A(im_size)
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        ax = torch.real(
            torch.fft.ifft2(FFT_H * FFT_x, dim = (-2,-1))
        )
        return ax
    
    def ATx(self, x):
        im_size = x.shape[-1]
        FFT_HC = torch.conj(self.blur2A(im_size))
        FFT_x = torch.fft.fft2(x, dim = (-2,-1))
        atx = torch.real(
            torch.fft.ifft2(FFT_HC * FFT_x, dim = (-2,-1))
        )
        return atx
    
    def y(self, x:torch.Tensor)->torch.Tensor:
        out = self.Ax(x) + self.sigma_model * torch.randn_like(x)
        return out

# Half-Quadratic Splitting module       
class HQS_models:
    def __init__(
        self,
        model, 
        physic:physics_models, 
        diffusion_scheduler:DiffusionScheduler, 
    ) -> None:
        self.model = model
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
    
    def proxf_update(
        self, 
        y:torch.Tensor,
        x:torch.Tensor, 
        x_t:torch.Tensor, 
        rho:float, 
        t:int
    )-> torch.Tensor:
        sigma_t = extract_tensor(self.diffusion_scheduler.sigmas, t, x_t.shape)
        sqrt_alpha_comprod = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape)
    
        self.im_size = y.shape[-1]
        FFTy = torch.fft.fft2(y, dim = (-2,-1))
        FFT_H = self.physic.blur2A(self.im_size)
        FFT_HC = torch.conj(FFT_H)
        FFTx = torch.fft.fft2(x, dim=(-2,-1))
        FFTx_t = torch.fft.fft2(x_t, dim=(-2,-1))
        _mu = FFT_HC * FFTy / self.physic.sigma_model**2 + FFTx / rho**2 + FFTx_t  / (sigma_t**2 * sqrt_alpha_comprod)
       
        return torch.real(
            torch.fft.ifft2(
                _mu * self._precision(FFT_H, rho, sigma_t),
                dim = (-2,-1)
            )
        )
        
    def _precision(
        self,
        FFT_H:torch.Tensor, 
        rho:float, 
        sigma_t:float
        ) -> torch.Tensor:
        out =  (
            FFT_H**2 / self.physic.sigma_model**2 
            + 1 / sigma_t**2
            + 1 / rho**2 
        )
        
        return 1 / out
    
    def run_unfolded_loop(
        self, 
        y:torch.Tensor, 
        x:torch.Tensor, 
        x_t:torch.Tensor, 
        t:int, 
        max_iter:int = 3
    ) -> dict:
        B,C = x.shape[:2]
        half_t = t / 1
        half_t = half_t.long()
        rho = 1000#extract_tensor(self.diffusion_scheduler.sigmas,half_t, x_t.shape)
        for _ in range(max_iter):
            # prox step
            z = self.proxf_update(y, x, x_t, rho, t)
            wandb.log({"min z":z.min(), "max z":z.max()})
            
            #z = torch.clamp(z, 0.0, 1.0)
            sigma_t = extract_tensor(self.diffusion_scheduler.sigmas, half_t, x_t.shape)
            sqrt_alpha_comprod = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape)
            eps = self.model(z, half_t)
            eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_t.shape[2:]) else (eps, eps)
            
            x_pred = self.diffusion_scheduler.predict_xstart_from_eps(z, half_t, eps)
            wandb.log({"min eps":eps.min(), "max eps":eps.max()})
            wandb.log({"min x_pred":x_pred.min(), "max x_pred":x_pred.max()})
            # wandb.log({"min x_1":x_1.min(), "max x_1":x_1.max()})
        
        return {"u":x_pred,"xstart_pred":x_pred, "aux":z, "eps_pred":eps, "noisy_image":inverse_image_transform(x_pred),}

class HQS_models_new:
    def __init__(
        self,
        model, 
        physic:physics_models, 
        diffusion_scheduler:DiffusionScheduler, 
    ) -> None:
        self.model = model
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
    
    def proxf_update(
        self, 
        y:torch.Tensor,
        x:torch.Tensor, 
        rho:float, 
    )-> torch.Tensor:    
        self.im_size = y.shape[-1]
        # y = torch.clamp(y, 0, 1)
        # x = torch.clamp(x, 0, 1)
        FFTy = torch.fft.fft2(y, dim = (-2,-1))
        FFT_H = self.physic.blur2A(self.im_size)
        FFT_HC = torch.conj(FFT_H)
        FFTx = torch.fft.fft2(x, dim=(-2,-1))
        _mu = FFT_HC * FFTy / self.physic.sigma_model**2 + FFTx / rho**2 
       
        return torch.real(
            torch.fft.ifft2(
                _mu * self._precision(FFT_H, rho),
                dim = (-2,-1)
            )
        )
        
    def _precision(
        self,
        FFT_H:torch.Tensor, 
        rho:float, 
        ) -> torch.Tensor:
        out =  (
            FFT_H**2 / self.physic.sigma_model**2 
            + 1 / rho 
        )
        
        return 1 / out
    
    def run_unfolded_loop(
        self, 
        y:torch.Tensor, 
        x:torch.Tensor, 
        x_t:torch.Tensor, 
        t:int, 
        max_iter:int = 3
    ) -> dict:
        B,C = x.shape[:2]
        t_rho = t 
        t_rho = t_rho.long()
        rho = extract_tensor(self.diffusion_scheduler.sigmas, t_rho, x.shape) + 100000
        #rho = 1000#rho / (rho.max().sqrt())
        
        half_t = t
        half_t = half_t.long()
        sigma_t = extract_tensor(self.diffusion_scheduler.sigmas, half_t, x_t.shape)
        sqrt_alpha_comprod = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, x_t.shape)
        
        # unfolded loop
        for _ in range(max_iter):
            
            # prox step
            z = self.proxf_update(y, x, rho)
            wandb.log({"min z":z.min(), "max z":z.max()})
            z = torch.clamp(z, 0, 1)
            
            # noisy image 
            Sigma_inv = (sigma_t**2 * rho) / (sigma_t**2+rho)
            _mu_z = z/rho + x_t / (sigma_t**2 * sqrt_alpha_comprod)
            u = Sigma_inv * _mu_z
            
            # noise prediction         
            eps = self.model(u, t_rho)
            eps, _ = torch.split(eps, C, dim=1) if eps.shape == (B, C * 2, *x_t.shape[2:]) else (eps, eps)
            x_1 = self.diffusion_scheduler.predict_xstart_from_eps(u, t_rho, eps)
            x_pred = u - Sigma_inv.sqrt() * eps
            x = inverse_image_transform(x_pred)
            
            ##############
            ##############
            eps0 = self.model(x_t, half_t)
            eps0, _ = torch.split(eps0, C, dim=1) if eps0.shape == (B, C * 2, *x_t.shape[2:]) else (eps0, eps0)
            x_10 = self.diffusion_scheduler.predict_xstart_from_eps(x_t, half_t, eps0)
            ##############
            ##############
            
            wandb.log({"min eps":eps.min(), "max eps":eps.max()})
            wandb.log({"min x_":x_pred.min(), "max x_":x_pred.max()})
            wandb.log({"min x_1":x_1.min(), "max x_1":x_1.max()})
            wandb.log({"min u":u.min(), "max u":u.max()})
            wandb.log({"min x10":x_10.min(), "max x10":x_10.max()})
            wandb.log({"min xt":x_t.min(), "max xt":x_t.max()})
        
        return {"u":u,"xstart_pred":x_pred, "aux":z, "eps_pred":eps, "noisy_image":inverse_image_transform(x_10),}
           
# trainer model
class Trainer:
    def __init__(
        self,
        model,
        diffusion_scheduler:DiffusionScheduler,
        physic:physics_models,
        trainloader,
        testloader,
        logger,
        device:str,
        args,
        max_unfolded_iter:int = 1,
        ) -> None:
        self.args = args
        self.logger = logger
        self.diffusion_scheduler = diffusion_scheduler
        self.physic = physic
        self.max_unfolded_iter = max_unfolded_iter
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        
        # Accelerator: DDP
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        
        # set training
        self._set_training()
        self.metrics = Metrics(self.device)
        
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
        self.hqs_model = HQS_models(
            self.model,
            self.physic, 
            self.diffusion_scheduler, 
        )
    
    def training(self, epochs):
        """_summary_
        """
        dir_w = "../sharedscratch/Results/wandb"
        wandb.init(
                # set the wandb project where this run will be logged
                project="Training Unfolded Conditional Diffusion Models",
                dir=dir_w,

                # track hyperparameters and run metadata
                config={
                "learning_rate": self.args.learning_rate,
                "architecture": "Deep Unfolding",
                "Structure": "HQS",
                "dataset": "FFHQ - 256",
                "epochs": epochs,
                "dir": dir_w,
                }
            )
        
        for epoch in range(epochs):
            # training mode
            self.logger.info(f"Epoch ==== >> {epoch}")
            self.model.train()
            idx = 0
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
                    sqrt_alpha_cumprod_t = extract_tensor(self.diffusion_scheduler.sqrt_alphas_cumprod, t, y.shape)
                   
                    # noisy image in [0,1] from x_t in [-l,l]
                    xt_unscale = x_t - sqrt_alpha_cumprod_t * (batch_0 - 1)
                    
                    # initialisation unfolded model
                    x_init = (
                        sigmas_t**2 / (sigmas_t**2 + self.physic.sigma_model**2) * xt_unscale + 
                        self.physic.sigma_model**2 / (sigmas_t**2 + self.physic.sigma_model**2) * y
                    )
                    # unfolded model
                    out = self.hqs_model.run_unfolded_loop( y, y, x_t, t, max_iter = self.max_unfolded_iter)
                    xstart_pred = out["xstart_pred"]
                    eps_pred = out["eps_pred"]
                    aux = out["aux"]
                    u_noisy = out["noisy_image"]
                    u= out["u"]
                    
                    # loss
                    loss = nn.MSELoss()(eps_pred, noise_target).mean()
                    if self.args.use_consistancy:
                        loss += consistancy_loss(xstart_pred, y, self.physic)
                    self.accelerator.backward(loss)        

                    self.optimizer.step()
                    self.scheduler.step() 
                    self.model.zero_grad()
            
                with torch.no_grad():
                    loss_val = loss.clone().detach().cpu().mean().item()
                    self.logger.info(f"Loss: {loss_val}")            
                    wandb.log({"loss": loss_val}) if self.args.use_wandb else None
                    
                    if self.args.use_wandb:
                        wandb.log(
                            {
                            "mse": self.metrics.mse_function(
                                        inverse_image_transform(xstart_pred.detach().cpu()), 
                                        batch_0.detach().cpu()
                                    ).item(), 
                            "psnr": self.metrics.psnr_function(
                                        inverse_image_transform(xstart_pred.detach().cpu()), 
                                        batch_0.detach().cpu()
                                    ).item()
                            }
                            )
                        
                    # reporting some images 
                    
                    if ii % 100 == 0:
                        if self.args.use_wandb:
                            xstart_pred_wdb = wandb.Image(
                                    get_batch_rgb_from_batch_tensor(
                                        inverse_image_transform(xstart_pred)
                                    ), 
                                    caption=f"x0_hat {idx}_{epoch}", 
                                    file_type="png"
                                )
                            batch_wdb = wandb.Image(
                                    get_batch_rgb_from_batch_tensor(batch_0), 
                                    caption=f"x0 {idx}_{epoch}", 
                                    file_type="png"
                                )
                            ys_wdb = wandb.Image(
                                    get_batch_rgb_from_batch_tensor(y), 
                                    caption=f"y {idx}_{epoch}", 
                                    file_type="png"
                                )
                            xt_wdb = wandb.Image(
                                    get_batch_rgb_from_batch_tensor(x_t), 
                                    caption=f"xt {idx}_{epoch}", 
                                    file_type="png"
                                )
                            aux_wdb = wandb.Image(
                                    get_batch_rgb_from_batch_tensor(aux), 
                                    caption=f"z {idx}_{epoch}", 
                                    file_type="png"
                                )
                            u_noisy_wdb = wandb.Image(
                                    get_batch_rgb_from_batch_tensor(u_noisy), 
                                    caption=f"u noisy {idx}_{epoch}", 
                                    file_type="png"
                                )
                            u_wdb = wandb.Image(
                                    get_batch_rgb_from_batch_tensor(u), 
                                    caption=f"u  {idx}_{epoch}", 
                                    file_type="png"
                                )
                            wandb.log(
                                {
                                    "ref image": batch_wdb, 
                                    "blurred":ys_wdb, 
                                    "xt":xt_wdb, 
                                    "U noisy":u_noisy_wdb,
                                    "x predict": xstart_pred_wdb, 
                                    "Auxiliary":aux_wdb,
                                    "u":u_wdb,
                                }
                            )
                            
                            idx += 1
                    #########################################################
                    if ii == 100:
                        pass
                        
                    end_time = time.time()  
                    ellapsed = end_time - t_start       
                    self.logger.info(f"Elapsed time per epoch: {ellapsed}") 

            torch.cuda.empty_cache() 
            
            # Validation 
            if 1 == 2:
                with torch.no_grad():
                    self.model.eval()
                    im = 0
                    for ii, batch_val in enumerate(self.testloader):
                        
                        seq, progress_seq = self.diffusion_scheduler.get_seq_progress_seq(100)
                        
                        # observation y
                        y = self.physic.y(batch_val)
                        x = torch.randn_like(batch_val)
                        progress_img = []
                        
                        for ii in range(len(seq)):
                            tii = seq[ii]
                            # unfolding: xstrat_pred, eps_pred, auxiliary variable
                            sigmas_tii = extract_tensor(self.diffusion_scheduler.sigmas, tii, y.shape)
                            x_init = (
                                sigmas_tii / (sigmas_t + self.physic.sigma_model) * x + 
                                self.physic.sigma_model / (sigmas_tii + self.physic.sigma_model) * y
                            )
                            rho =  sigmas_t / 2
                            out_val = self.hqs_model.run_unfolded_loop( y, x_init, x, tii, rho, max_iter = self.max_unfolded_iter)
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
                            x_0 = inverse_image_transform(x)
                            
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
                            y_wdb = wandb.Image(
                                get_rgb_from_tensor(y), 
                                caption=f"observation {im}", 
                                file_type="png"
                            )
                            xy = [y_wdb, x0_wdb]
                            wandb.log({"blurred and predict":xy})
        
    # - save the checkpoint model   
    def save(self, epoch):
        checkpoint_dir = os.makedirs(
            bf.join(self.args.save_checkpoint_dir, self.args.date),
            exist_ok=True
        )
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
                    }, f)

        del unwrap_model
            
def get_rgb_from_tensor(x):
    x0_np = x.detach().cpu().numpy()       #[0,1]
    x0_np = np.squeeze(x0_np)
    if x0_np.ndim == 3:
        x0_np = np.transpose(x0_np, (1, 2, 0))
    return x0_np
def get_batch_rgb_from_batch_tensor(x):
    L = [get_rgb_from_tensor(xi) for xi in x]
    out= cv2.hconcat(L)*255.
    return out
                    
def consistancy_loss(
    x:torch.Tensor,
    y:torch.Tensor,
    physic: physics_models,
) -> torch.Tensor:
    device = x.device
    # Create a list of kernels and images
    mse_per_sample = []
    for i in range(y.shape[0]):
        x_i = inverse_image_transform(x[i,...])
        Ax_i = physic.Ax(x_i).squeeze()
        mse_per_sample.append(nn.MSELoss()(Ax_i, y[i]).mean().item())
    mse_val = torch.tensor(mse_per_sample).to(device)
    loss =  mse_val #/ (2*physic.sigma_model**2)  
    return loss.mean()
      
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
    


    
    