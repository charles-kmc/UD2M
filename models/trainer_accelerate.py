import numpy as np
import time
import os
import wandb
import blobfile as bf 
import torch  
import torch.nn as nn 
import torch.distributed as dist
from torch.optim import AdamW
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from accelerate import Accelerator

from . import dist_util
from .resample import LossAwareSampler, UniformSampler
from degradation_model.utils_deblurs import deb_batch_data_solution
from utils.utils import Metrics, inverse_image_transform, image_transform
from models.ema import EMA
from utils_lora.lora_parametrisation import LoRaParametrisation
from guided_diffusion.unet import TimestepEmbedSequential

from DPIR.dpir_models import DPIR_deb, dpir_module, dpir_solver

from memory_profiler import profile 
import psutil
from torchviz import make_dot 


INITIAL_LOG_LOSS_SCALE = 20.0

class Trainer_accelerate:
    def __init__(
            self,
            model,
            ema_model,
            diffusion,
            dataloader,
            logger,
            gradient_accumulation_steps = 1,
            gradient_accumulation = True,
            clip_value = 0.5,
            max_grad_norm = 1e0,
            batch_size = 32, 
            ema_rate = 0.99, 
            log_interval = 100,
            save_interval = 100, 
            lr = 1e-3,
            save_checkpoint_dir = "",
            resume_checkpoint = "",
            save_results = "",
            learn_alpha = False,
            schedule_sampler = None,
            weight_decay = 0.0,
            lr_anneal_epochs = 0,
            cons_loss = True,
            problem_type = "deblur",
            solver = "dpir",
        ):
        
        self.logger = logger
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_accumulation = gradient_accumulation
        self.lr= lr
        self.clip_value = clip_value
        self.max_grad_norm = max_grad_norm
        self.ema_rate = ema_rate
        self.save_checkpoint_dir = save_checkpoint_dir
        self.save_results = save_results
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_epochs = lr_anneal_epochs
        self.cons_loss = cons_loss
        self.learn_alpha = learn_alpha
        self.epoch = 0
        self.resume_epoch = 0
        
        self.task = problem_type
        
        # DPIR
        self.solver = solver
        drunet = 0
        self.model_name = "drunet_color" if drunet else "ircnn_color"  
        net_path = "/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo"
        self.dpir = DPIR_deb(model_name = self.model_name,pretrained_pth = net_path)
        self.iter_num = 1
        self.alpha = 0.01
        
        self.model = self.model.to(dist_util.dev())
        
        # load parameters of the model for training and synchronise
        self._load_and_sync_parameters()
        
        # Accelerator: DDP
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        
        # parameters
        model_parameters(self.model, self.logger)
        
        # learnable parameters
        self.model_params_learnable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        # optimizer
        self.optimizer = AdamW(self.model_params_learnable, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(100 * 0.2), gamma=0.1)
        
        # ema_model
        self.ema = EMA(ema_model)
        if self.resume_epoch:
            self._load_ema_parameters()
            self._load_optimizer()
        
        # pass training object into accelerator
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
                                                                                                self.model, 
                                                                                                self.optimizer, 
                                                                                                self.dataloader, 
                                                                                                self.scheduler,                                                                                                 )
        
        
        # 
        self.metrics = Metrics(dist_util.dev())
        
        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_epoch = parse_resume_epoch_from_filename(resume_checkpoint)
            if self.accelerator.is_main_process:
                self.logger.info(f"Resuming from checkpoint {resume_checkpoint}")
                
                # loading checkpoints
                checkpoint = dist_util.load_state_dict(resume_checkpoint)
                
                # Loading weights
                self.model.load_state_dict(checkpoint["model_state_dict"], map_location=dist_util.dev())
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"], map_location=dist_util.dev())
                
        # sychronise parameters of the model
        dist_util.sync_params(self.model.parameters())
        
    # load ema checkpoints
    def _load_ema_parameters(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint       
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.task, self.resume_epoch, self.ema_rate)
        if ema_checkpoint:
            if self.accelerator.is_main_process:
                self.logger.info(f"Loading EMA parameters from {ema_checkpoint}")
                
                # loading checkpoints
                checkpoint = dist_util.load_state_dict(ema_checkpoint)
                
                # Loading weights
                self.ema.ema_model.load_state_dict(checkpoint["model_state_dict"], map_location=dist_util.dev())
        
        # sychronise parameters of the model
        dist_util.sync_params(self.ema.ema_model.parameters())
    
    def _load_optimizer(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_epoch = parse_resume_epoch_from_filename(resume_checkpoint)
            if self.accelerator.is_main_process:
                self.logger.info(f"Resuming from checkpoint {resume_checkpoint}")
                
                # loading checkpoints
                checkpoint = dist_util.load_state_dict(resume_checkpoint)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"], map_location=dist_util.dev())
        
    # degradation model
    def _degradation_model(self, batch_y, blur_kernel_op):
        if self.solver == "tikhonov":
            self.ad_name = self.solver + "_gauss"
            if self.task == "deblur":
                batch_cond = deb_batch_data_solution(batch_y, blur_kernel_op, self.device, alpha = self.alpha)
                return batch_cond
            elif self.task == "inp":
                raise NotImplementedError 
            elif self.task == "sr":
                raise NotImplementedError
        elif self.solver == "dpir":
            self.ad_name = self.solver+"_"+self.model_name
            if self.task == "deblur": 
                batch_cond = dpir_module(batch_y, blur_kernel_op, self.dpir, self.device, iter_num=self.iter_num)
                return batch_cond
            elif self.task == "inp":
                raise NotImplementedError 
            elif self.task == "sr":
                raise NotImplementedError
        
    # Training loop
    #@profile(stream=open('memory_profiler_output.log', 'w+'))
    def run_training_loop(self, epochs):
        """_summary_
        """
        wdir = "/users/cmk2000/cmk2000/Deep learning models/LLMs/logs/Conditional-Diffusion-Models-for-IVP/wandb"
        os.makedirs(wdir, exist_ok=True)
        wandb.init(
                # set the wandb project where this run will be logged
                project="Training conditinal diffusion models",
                dir="/users/cmk2000/cmk2000/Deep learning models/LLMs/logs/Conditional-Diffusion-Models-for-IVP/wandb",

                # track hyperparameters and run metadata
                config={
                "learning_rate": self.lr,
                "architecture": "Deep Cnn",
                "dataset": "ImageNet - 256",
                "epochs": epochs,
                }
            )
        # path
        est_path = os.path.join(self.save_results, "est")
        refs_path = os.path.join(self.save_results, "refs")
        degrads_path = os.path.join(self.save_results, "degrads")
        sample_path = os.path.join(self.save_results, "samples")
        est_solver_path = os.path.join(self.save_results, "est_solvers")
        os.makedirs(refs_path, exist_ok=True)
        os.makedirs(est_path, exist_ok=True)
        os.makedirs(degrads_path, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)
        os.makedirs(est_solver_path, exist_ok=True)
        
        for epoch in range(epochs):
            # training mode
            self.logger.info(f"Epoch ==== >> {epoch}")
            self.model.train()
            for ii, (batch_data, batch_y, blur_kernel_op) in enumerate(self.dataloader):
                
                with torch.no_grad():
                    # tracking monitor memory
                    process = psutil.Process(os.getpid())
                    self.logger.info(f"-----------------------------------------------------------")
                    self.logger.info(f"-----------------------------------------------------------")
                    self.logger.info(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")
                    self.logger.info(f"-----------------------------------------------------------")
                    self.logger.info(f"-----------------------------------------------------------\n")
                    

                    # start epoch run
                    t_start = time.time()
                    self.logger.info(f"Epoch ==== >> {epoch}\t|\t batch ==== >> {ii}\n") 
                    self.ii = ii
                with self.accelerator.accumulate(self.model):       
                    
                    # conditioning
                    batch_cond =   self._degradation_model(batch_y, blur_kernel_op)
                    
                    # epoch step through the model
                    self.run_epoch(batch_data, batch_cond, epoch)
                    
                    
                with torch.no_grad():   
                    if 1 == 1:
                        wandb.log(
                            {
                            "mse": self.metrics.mse_function(
                                        inverse_image_transform(self.pred_xstart.cpu()), 
                                        inverse_image_transform(batch_data.detach().cpu())
                                    ).item(), 
                            "psnr": self.metrics.psnr_function(
                                        inverse_image_transform(self.pred_xstart.cpu()), 
                                        inverse_image_transform(batch_data.detach().cpu())
                                    ).item()
                            })
                        # reporting some images 
                        
                    if ii % 100 == 0:
                        save_image(inverse_image_transform(self.pred_xstart.cpu()), os.path.join(est_path, f"est_{epoch}.png"))
                        save_image(inverse_image_transform(batch_data.detach().cpu()), os.path.join(refs_path, f"ref_{epoch}.png"))
                        save_image(batch_y.detach().cpu(), os.path.join(degrads_path, f"degrad_{epoch}.png"))
                        save_image(batch_cond.detach().cpu(), os.path.join(est_solver_path, f"est_{self.ad_name}_{epoch}.png"))
                        save_image(inverse_image_transform(self.x_t.cpu()), os.path.join(sample_path, f"sample_{epoch}.png"))
                        
                
                    end_time = time.time()  
                    ellapsed = end_time - t_start       
                    self.logger.info(f"Elapsed time per epoch: {ellapsed}")  
                
                del batch_cond
                torch.cuda.empty_cache()
                
            # Save the last checkpoint if it wasn't already saved.
            with torch.no_grad():
                if epoch % self.save_interval == 0:
                    self.accelerator.wait_for_everyone()
                    self.save(epoch)
                else:
                    self.accelerator.wait_for_everyone()
                    self.save(0)
        
    # epoch through the model      
    #@profile(stream=open('memory_profiler_output.log', 'w+')) 
    def run_epoch(self, batch_data, batch_cond, epoch):
        self.run_forward_backward(batch_data, batch_cond)
        self.optimizer.step()
        ema_update = True
        if ema_update:
            self.ema.update(self.accelerator.unwrap_model(self.model), self.ema_rate)
        self.scheduler.step() #_anneal_lr(epoch)
        self.model.zero_grad()
        with torch.no_grad():
            for _, module in self.model.named_modules():
                if isinstance(module, LoRaParametrisation):
                    module.lora_A.zero_()
                    module.lora_B.zero_()
        self.log_epoch(epoch, len(batch_data))
    
    def run_forward_backward(self, batch_data, batch_cond):    
        #zero_grad(self.model_params_learnable)     
        # weight from diffusion sampler    
        t, weights = self.schedule_sampler.sample(batch_data.shape[0], dist_util.dev())
        B,C = batch_data.shape[:2]
        
        # get noise
        noise_target = torch.randn_like(batch_data)
        # sample noisy x_t
        self.x_t = self.diffusion.q_sample(batch_data, t, noise=noise_target)
    
        noise_pred  = self.model(self.x_t, t, batch_cond)
        if noise_pred.shape == (B, C * 2, *self.x_t.shape[2:]):
            noise_pred, _ = torch.split(noise_pred, C, dim=1)
        else:
            pass
        
        # get x_0 from noise_pred
        self.pred_xstart = self.diffusion._predict_xstart_from_eps(self.x_t, t, noise_pred)
        
        # loss
        if self.cons_loss:   
            loss = self.model.consistency_loss(self.pred_xstart, self.batch_y, self.blur_kernel_op)         
        else:
            loss = 0.0
        loss += nn.MSELoss()(noise_pred, noise_target)
         
        with torch.no_grad():
            loss_val = loss*weights 
            self.logger.info(f"Loss: {loss_val.detach().cpu().mean().item()}")            
            # monitor loss with wandb
            wandb.log({"loss": loss_val.detach().cpu().mean().item()})
            
        # back-propagation: here, gradients are computed
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_value_(self.model_params_learnable,  self.clip_value)
            self.accelerator.clip_grad_norm_(self.model_params_learnable,   self.max_grad_norm)
        
    # - log the epoch
    def log_epoch(self, epoch, samples_size):
        self.logger.info(f"epoch:  {epoch + self.resume_epoch} \t batch: {self.ii}\t samples: {samples_size}")
    
    # - save the checkpoint model   
    def save(self, epoch):
        def save_checkpoint(rate, model):
            try:
                unwrap_model = self.accelerator.unwrap_model(model)
            except:
                unwrap_model = model
                
            if self.accelerator.is_main_process:
                self.logger.info(f"Saving EMA checkpoint for rate {rate}...")
                if not rate:
                    filename = f"{self.task}_model_{(epoch+self.resume_epoch):03d}.pt"
                else:
                    filename = f"{self.task}_ema_model_rate_{rate}_{(epoch+self.resume_epoch):03d}.pt"
                    
                filepath = bf.join(self.save_checkpoint_dir, filename)
                with bf.BlobFile(filepath, "wb") as f:
                    torch.save({
                            'model_state_dict': unwrap_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'epoch': epoch,
                        }, f)

            del unwrap_model
            del filename
            del model
        save_checkpoint(0, self.model)
        save_checkpoint(self.ema_rate, self.ema.ema_model)
        dist.barrier()

     
# - log the loss            
def log_loss_dict(diffusion, ts, losses, logger):
    for key, values in losses.items():
        logger.info(f"{key}:  {values.mean().item():.4f}")
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            #logger.info(f"{key}_q{quartile}: {sub_loss:.4f}")   


# - find the resume checkpoint      
def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


# - parse the resume epoch from the filename
def parse_resume_epoch_from_filename(filename):
    """We assume the name of the model as the following format: /path/to/modelNNNNN.pt, where NNNNN is the checkpoint,s number of epochs.
    """
    split_name = filename.split("model")
    if len(split_name) <2:
        return 0
    
    split1 = split_name[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
    
    
# - find the ema checkpoint
def find_ema_checkpoint(main_checkpoint, task, epoch, rate):
    if main_checkpoint is None:
        return None
    filename = f"{task}_ema_{rate}_{(epoch):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


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
   
    
# reset to zero
def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def params_to_state_dict(model, params):
    state_dict = model.state_dict()
    for i, (name, _) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = params[i]
    return state_dict


def optimizer_step(optimizer: torch.optim.Optimizer, params, logger):
        return optimize_normal(optimizer, params, logger)


def optimize_normal(optimizer, params, logger):
    # grad_norm, param_norm = compute_norms(params)
    # logger.info(f"grad_norm: {grad_norm:.4f} ----- param_norm: {param_norm:.4f}")
    optimizer.step()
    return True


def compute_norms(params, grad_scale=1.0):
    grad_norm = 0.0
    param_norm = 0.0
    for p in params:
        with torch.no_grad():
            param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
            if p.grad is not None:
                grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
    return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


def plot_metric(value, name):
    fig1 = plt.figure()
    plt.plot(torch.arange(len(torch.cat(value))), torch.cat(value))
    plt.savefig(f"{name}.png") 
        
        