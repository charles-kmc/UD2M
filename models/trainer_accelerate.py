import numpy as np#type:ignore
import time
import copy
import functools
import os
import wandb
import blobfile as bf #type:ignore
import torch  #type: ignore
import torch.distributed as dist#type: ignore
from torch.nn.parallel.distributed import DistributedDataParallel as DDP#type: ignore
from torch.optim import AdamW#type: ignore
from torchvision.utils import save_image#type: ignore
import matplotlib.pyplot as plt#type: ignore

from accelerate import Accelerator#type:ignore

from . import dist_util
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from degradation_model.utils_deblurs import deb_batch_data_solution
from utils.utils import Metrics, inverse_image_transform, image_transform
from models.ema import EMA

from memory_profiler import profile #type:ignore
import psutil


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
            learn_alpha = True,
            schedule_sampler = None,
            weight_decay = 0.0,
            lr_anneal_epochs = 0,
            consistency_loss = True,
            problem_type = "deblur",
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
        self.consistency_loss = consistency_loss
        self.learn_alpha = learn_alpha
        self.epoch = 0
        self.resume_epoch = 0
        
        self.task = problem_type
        #self.global_batch = self.batch_size * dist.get_world_size()
        
        self.syn_cuda = torch.cuda.is_available()
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
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(100 * 0.8))
        
        # ema_model
        self.ema = EMA(self.model, ema_model)
        
        # ema model 
        if self.resume_epoch:
            self._load_ema_parameters()
            self._load_optimizer()
        
        # pass training object into accelerator
        self.model, self.optimizer, self.dataloader, self.scheduler, self.anneal_lr = self.accelerator.prepare(
                                                                                                self.model, 
                                                                                                self.optimizer, 
                                                                                                self.dataloader, 
                                                                                                self.scheduler, 
                                                                                                self._anneal_lr
                                                                                                )
        
        # 
        self.global_batch = self.batch_size * dist.get_world_size()  
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
                self.ema.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], map_location=dist_util.dev())
        
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
    def _degradation_model(self, batch_y, blur_kernel_op, device, alpha = 0.01):
        if self.task == "deblur":
            batch_sol_Tk = deb_batch_data_solution(batch_y, blur_kernel_op, device, alpha = alpha)
            return batch_sol_Tk
        elif self.task == "inp":
            raise NotImplementedError 
        elif self.task == "sr":
            raise NotImplementedError
           
    # Training loop
    @profile(stream=open('memory_profiler_output.log', 'w+'))
    def run_training_loop(self, epochs):
        """_summary_
        """
        wandb.init(
                # set the wandb project where this run will be logged
                project="Training conditinal diffusion models",

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
        os.makedirs(refs_path, exist_ok=True)
        os.makedirs(est_path, exist_ok=True)
        os.makedirs(degrads_path, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)
        
        for epoch in range(epochs):
            # training mode
            self.logger.info(f"Epoch ==== >> {epoch}")
            self.model.train()
            for ii, (batch_data, batch_y, blur_kernel_op) in enumerate(self.dataloader):
                
                with torch.no_grad():
                    # tracking monitor memory
                    process = psutil.Process(os.getpid())
                    self.logger.info(f"\n")
                    self.logger.info(f"-----------------------------------------------------------")
                    self.logger.info(f"-----------------------------------------------------------")
                    self.logger.info(f"Memory usage: {process.memory_info().rss / 1024 ** 2} MB")
                    self.logger.info(f"-----------------------------------------------------------")
                    self.logger.info(f"-----------------------------------------------------------")
                    self.logger.info(f"\n")
                    

                    # start epoch run
                    t_start = time.time()
                    self.logger.info(f"Epoch ==== >> {epoch}\t|\t batch ==== >> {ii}\n") 
                    self.ii = ii
                with self.accelerator.accumulate(self.model):       
                    # Tikhonov data solution for consistency
                    if self.learn_alpha:
                        lr_alp = self.model.alpha
                        batch_sol_Tk =   self._degradation_model(batch_y, blur_kernel_op, dist_util.dev(), alpha = lr_alp)
                    else:
                        batch_sol_Tk =   self._degradation_model(batch_y, blur_kernel_op, dist_util.dev())
                    # epoch step through the model
                    self.run_epoch(batch_data, batch_sol_Tk, batch_y, blur_kernel_op, epoch)
                    # epoch evaluation
                    
                with torch.no_grad():   
                    if 1 == 1:
                        wandb.log(
                            {
                            "psnr": self.metrics.mse_function(
                                        inverse_image_transform(self.x_pred.cpu()), 
                                        inverse_image_transform(batch_data.detach().cpu())
                                    ).item(), 
                            "mse": self.metrics.psnr_function(
                                        inverse_image_transform(self.x_pred.cpu()), 
                                        inverse_image_transform(batch_data.detach().cpu())
                                    ).item()
                            }
                            )
                        # reporting some images 
                        
                    if ii % 100 == 0:
                        save_image(inverse_image_transform(self.x_pred.cpu()), os.path.join(est_path, f"est_{epoch}.png"))
                        save_image(inverse_image_transform(batch_data.detach().cpu()), os.path.join(refs_path, f"ref_{epoch}.png"))
                        save_image(batch_y.detach().cpu(), os.path.join(degrads_path, f"degrad_{epoch}.png"))
                        save_image(self.x_t.cpu(), os.path.join(sample_path, f"sample_{epoch}.png"))
                        
                
                    end_time = time.time()  
                    ellapsed = end_time - t_start       
                    self.logger.info(f"Elapsed time per epoch: {ellapsed}")  
                
                del batch_sol_Tk
                torch.cuda.empty_cache()
                
            # Save the last checkpoint if it wasn't already saved.
            with torch.no_grad():
                if epoch % self.save_interval != 0:
                    self.accelerator.wait_for_everyone()
                    self.save(epoch)
        
    # epoch through the model      
    #@profile(stream=open('memory_profiler_output.log', 'w+')) 
    def run_epoch(self, batch_data, batch_sol_Tk, batch_y, blur_kernel_op, epoch):
        self.run_forward_backward(batch_data, batch_sol_Tk, batch_y, blur_kernel_op)
        ema_update = True
        self.optimizer.step()
        if ema_update:
            self.ema.update(
                        self.accelerator.unwrap_model(self.model), 
                        self.ema_rate
                        )
        
        # set grad to zero
        zero_grad(self.model_params_learnable)
        
        # scheduler    
        self.scheduler.step() #_anneal_lr(epoch)
        self.log_epoch(epoch, len(batch_data))
    
    # forward and backward function
    #@profile(stream=open('memory_profiler_output.log', 'w+'))
    def run_forward_backward(self, batch_data, batch_sol_Tk, batch_y, blur_kernel_op):        
        # weight from diffusion sampler    
        t, weights = self.schedule_sampler.sample(batch_data.shape[0], dist_util.dev())
        
        # - compute the losses
        batch_sol_Tk.requires_grad_(False)
        # model_y = lambda x, t: self.model(x, t, batch_sol_Tk)
        compute_losses = functools.partial(
                    self.diffusion.losses,   
                    self.model,
                    batch_data,
                    t,
                    batch_sol_Tk
                ) 
        
        # computing losses
        losses_x_pred = compute_losses()
        
        # get the predicted image
        x_pred = losses_x_pred["pred_xstart"]
        # consistency loss
        if self.consistency_loss:   
            batch_y.requires_grad_(False)
            compute_losses_consis = functools.partial(
                self.model.consistency_loss,
                    x_pred, 
                    batch_y, 
                    blur_kernel_op
                ) 
            loss = compute_losses_consis()
        else:
            loss = 0.0
            
        loss += (losses_x_pred["mse"]*weights).mean()
        # log_loss_dict(
        #     self.diffusion, t, {"loss": losses_x_pred["loss"]*weights}, self.logger
        # )
        with torch.no_grad():
            loss_val = losses_x_pred["mse"].detach().cpu()*weights.cpu()
            self.logger.info(f"Loss: {loss_val.mean().item()}")            
            # monitor loss with wandb
            wandb.log({"loss": loss_val.mean().numpy()})
        
        # back-propagation: here, gradients are computed
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_value_(self.model_params_learnable,  self.clip_value)
            # self.accelerator.clip_grad_norm_(self.model_params_learnable,   self.max_grad_norm)
        
        self.x_pred = x_pred.detach()
        self.x_t = x_t.detach()
    
    # - anneal the learning rate   
    def _anneal_lr(self, epoch):
        if not self.lr_anneal_epochs:
            return
        else:
            frac_done = (epoch + self.resume_epoch) / self.lr_anneal_epochs
            lr = self.lr * max(1e-20, 1.0 - frac_done)
            
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

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
    plt.savefig("{name}.png") 
        
        