import numpy as np#type:ignore
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

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from degradation_model.utils_deblurs import deb_batch_data_solution
from utils.utils import Metrics

INITIAL_LOG_LOSS_SCALE = 20.0

class Trainer:
    def __init__(
            self,
            model,
            diffusion,
            dataloader,
            clip_value = 0.5,
            max_grad_norm = 1e0,
            batch_size = 32, 
            microbatch = 0, 
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
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr= lr
        self.clip_value = clip_value
        self.max_grad_norm = max_grad_norm
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        )
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
        
        
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        
        # parameters
        model_parameters(self.model)
        
        # learnable parameters
        self.model_params_learnable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        # optimizer
        self.optimizer = AdamW(self.model_params_learnable, lr=self.lr, weight_decay=self.weight_decay)
        
        # pass training object into accelerator
        self.model, self.optimizer, self.dataloader, self.anneal_lr = self.accelerator.prepare(model, self.optimizer, dataloader, self._anneal_lr)
        
        # ema model 
        if self.resume_epoch:
            # Model was resumed, either due to a restart or a checkpoint being specified at the command line.
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.model.parameters) for _ in range(len(self.ema_rate))
                ]
            
        self.global_batch = self.batch_size * dist.get_world_size()
                
        self.psnr_history = []
        self.mse_history = []
        self.loss_history = []
        self.metrics = Metrics(dist_util.dev())
        
        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_epoch = parse_resume_epoch_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.info(f"Resuming from checkpoint {resume_checkpoint}")
                
                # Loading weights
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        # sychronise parameters of the model
        dist_util.sync_params(self.model.parameters())
        
    # load ema checkpoints
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.model.parameters())     
        
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint       
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.task, self.resume_epoch, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.info(f"Loading EMA parameters from {ema_checkpoint}")
                state_dict = dist_util.load_state_dict(
                                ema_checkpoint, map_location=dist_util.dev()
                            )
                ema_params = self.model.state_dict_to_master_params(state_dict) 
        dist_util.sync_params(ema_params)
        return ema_params
    
        
    # load optimizer state
    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.json(
            bf.dirname(main_checkpoint), f"optimizer_{self.resume_epoch:06}.pt"
        )
        
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.optimizer.load_state_dict(state_dict)
    
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
        for epoch in range(epochs):
            # training mode
            print(f"Epoch ==== >> {epoch}")
            self.model.train()
            for ii, (batch_data, batch_y, blur_kernel_op) in enumerate(self.dataloader):
                print(f"\nEpoch ==== >> {epoch}\t|\t batch ==== >> {ii}\n") 
                self.ii = ii
                with self.accelerator.accumulate(self.model):       
                    
                    # Tikhonov data solution for consistency
                    if self.learn_alpha:
                        lr_alp = self.model.alpha
                        batch_sol_Tk = self._degradation_model(batch_y, blur_kernel_op, dist_util.dev(), alpha = lr_alp)
                    else:
                        batch_sol_Tk = self._degradation_model(batch_y, blur_kernel_op, dist_util.dev())
                    
                    # epoch step through the model
                    self.run_epoch(batch_data, batch_sol_Tk, batch_y, blur_kernel_op, epoch)
                    
                    # gethering information through logger
                    if epoch % self.log_interval == 0:
                        logger.dumpkvs()
                    
                    # save checkpoint    
                    if epoch % self.save_interval == 0:
                        self.save(epoch)
                    
                    # epoch evaluation
                    mse_val = self.metrics.mse_function(batch_data.detach().cpu(), self.x_pred.detach().cpu())
                    psnr_val = self.metrics.psnr_function(batch_data.detach().cpu(), self.x_pred.detach().cpu())
                    self.psnr_history.append(psnr_val)     
                    self.mse_history.append(mse_val)     
                    wandb.log({"psnr": psnr_val.numpy(), "mse": mse_val.numpy()})
                    
                # Save the last checkpoint if it wasn't already saved.
                if (epoch - 1) % self.save_interval != 0:
                    self.save(epoch)
                
                # reporting some images    
                if epoch % 20 == 0:
                    est_path = os.path.join(self.resume_results, "est")
                    refs_path = os.path.join(self.resume_results, "refs")
                    os.makedirs(refs_path, exist_ok=True)
                    os.makedirs(est_path, exist_ok=True)
                    save_image(self.x_pred.detach().cpu().numpy(), os.path.join(est_path, "est_{epoch}.png"))
                    save_image(self.batch_data.detach().cpu().numpy(), os.path.join(est_path, "ref_{epoch}.png"))
        
        # plotting: loss, psnr and mse
        fig1 = plt.figure()
        plt.plot(torch.arange(len(torch.cat(self.psnr_history))), torch.cat(self.psnr_history))
        plt.savefig("history_psnr.png") 
        
        fig1 = plt.figure()
        plt.plot(torch.arange(len(torch.cat(self.mse_history))), torch.cat(self.mse_history))
        plt.savefig("history_mse.png")
        
        fig1 = plt.figure()
        plt.plot(torch.arange(len(torch.cat(self.loss_history))), torch.cat(self.loss_history))
        plt.savefig("history_loss.png")   
        
    # epoch through the model       
    def run_epoch(self, batch_data, batch_sol_Tk, batch_y, blur_kernel_op, epoch):
        # pass data through forward and backward steps
        self.run_forward_backward(batch_data, batch_sol_Tk, batch_y, blur_kernel_op)
        
        # optimisation step
        ema_update = optimizer_set(self.optimizer, self.model_params_learnable)
        if ema_update:
            self._update_ema()
        
        # scheduler    
        self._anneal_lr(epoch)
        self.log_epoch(epoch)
    
    # forward and backward function
    def run_forward_backward(self, batch_data, batch_sol_Tk, batch_y, blur_kernel_op):
        zero_grad(self.model_params_learnable)
     
        t, weights = self.schedule_sampler.sample(batch_data.shape[0], dist_util.dev())
        
        # - compute the losses
        batch_sol_Tk.requires_grad_(False)
        model_y = lambda x, t: self.model(x, t, batch_sol_Tk)
        compute_losses = functools.partial(
                    self.diffusion.training_losses,   
                    model_y,
                    batch_data,
                    t
                ) 
        losses_x_pred = compute_losses()
        
        # get the predicted image
        x_pred = losses_x_pred["pred_xstart"]
        
        # consistency loss
        if self.consistency_loss:   
            batch_y.requires_grad_(False)
            losses_x_cons = self.model.consistency_loss(x_pred, batch_y, blur_kernel_op)
        
        loss = losses_x_cons
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses_x_pred["loss"].detach()
            )
            
        loss += (losses_x_pred["loss"]*weights).mean()
        log_loss_dict(
            self.diffusion, t, {"loss": losses_x_pred["loss"]*weights}
        )
        loss_val = losses_x_pred["loss"].detach().cpu()*weights.cpu()
        
        self.loss_history.append(loss_val.mean())
        wandb.log({"loss": loss_val.mean().numpy()})
        
        # back-propagation: here, gradients are computed
        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_value_(self.model_params_learnable,  self.clip_value)
            self.accelerator.clip_grad_norm_(self.model_params_learnable,   self.max_grad_norm)
        
        self.x_pred = x_pred
    
    # - update the ema model
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate = rate)
    
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
    def log_epoch(self, epoch):
        logger.logkv("epoch", epoch + self.resume_epoch)
        logger.logkv("batch", self.ii)
        logger.logkv("samples", (epoch + self.resume_epoch + 1) * self.global_batch)
    
    # - save the checkpoint model   
    def save(self, epoch):
        def save_checkpoint(rate, params):
            un_wrapped_model = self.accelerator.unwrap_model(self.model)
            
            state_dict = params_to_state_dict(un_wrapped_model, params)
            if self.accelerator.is_main_process:
                logger.log(f"Saving EMA checkpoint for rate {rate}...")
                if not rate:
                    filename = f"{self.task}_model_{(epoch+self.resume_epoch):06d}.pt"
                else:
                    filename = f"{self.task}_ema_{rate}_{(epoch+self.resume_epoch):06d}.pt"
                    filepath = bf.join(self.save_checkpoint_dir, filename)
                with bf.BlobFile(filepath, "wb") as f:
                    torch.save(state_dict, f)
                    
        save_checkpoint(0, self.model.parameters())
        
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
            
        if self.accelerator.is_main_process:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"{self.task}__opt__{(epoch+self.resume_epoch):06d}.pt"), "wb",
                ) as f:
                    torch.save(self.optimizer.state_dict(), f)   
        
        dist.barrier()
            
            
# - log the loss            
def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)   

# - get the blob log directory
def get_blob_logdir():
    """_summary_

    Returns:
        _type_: _description_
    """
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

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

def model_parameters(model):
    trained_params = 0
    Total_params = 0
    frozon_params = 0
    for p in model.parameters():
        Total_params += p.numel()
        if p.requires_grad:
            trained_params += p.numel()
        else:
            frozon_params += p.numel()
            
    print(f"Total parameters ====> {Total_params}")
    print(f"Total frozen parameters ====> {frozon_params}")
    print(f"Total trainable parameters ====> {trained_params}")
    
# reset to zero
def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def params_to_state_dict(model, params):
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = params[i]
    return state_dict


def optimizer_set(optimizer: torch.optim.Optimizer, params):
        return optimize_normal(optimizer, params)

def optimize_normal(optimizer, params):
    grad_norm, param_norm = compute_norms(params)
    logger.logkv_mean("grad_norm", grad_norm)
    logger.logkv_mean("param_norm", param_norm)
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