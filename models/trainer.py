import copy
import functools
import os

import blobfile as bf #type:ignore

import torch 
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from degradation_model.utils_deblurs import deb_batch_data_solution


INITIAL_LOG_LOSS_SCALE = 20.0

class Trainer:
    def __init__(
            self,
            model,
            diffusion,
            dataloader,
            batch_size = 32, 
            microbatch = 0, 
            ema_rate = 0.9999, 
            log_interval = 100,
            save_interval = 1000, 
            lr = 1e-3,
            save_checkpoint_dir = "",
            resume_checkpoint = "",
            learn_alpha = True,
            use_fp16 = False,
            fp16_scale_growth = 1e-3,
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
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        )
        self.save_checkpoint_dir = save_checkpoint_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_epochs = lr_anneal_epochs
        self.consistency_loss = consistency_loss
        self.problem_type = problem_type
        self.learn_alpha = learn_alpha
        self.epoch = 0
        self.resume_epoch = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        
        self.syn_cuda = torch.cuda.is_available()
        
        # load parameters of the model for training
        self._load_and_sync_parameters()
        
        # this allows the training to deal with different precision
        self.mp_trainer = MixedPrecisionTrainer(
                                    self.model, 
                                    use_fp16=self.use_fp16,
                                    fp16_scale_growth= self.fp16_scale_growth,
                                )
        
        # optimizer
        self.opt = AdamW(
            self.mp_trainer.master_params, 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # ema model 
        if self.resume_epoch:
            # Model was resumed, either due to a restart or a checkpoint being specified at the command line.
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params) for _ in range(len(self.ema_rate))
                ]
        
        # Distributed data parallel if cuda exists!!
        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                        self.model, 
                        device_ids=[dist_util.dev()],
                        output_device=dist_util.dev(),
                        find_unused_parameters=True,
                        bucket_cap_mb=128,
                        broadcast_buffers=False,
                        )
        else:
            if dist.get_world_size() > 1: 
                logger.warn(
                    "Distributed training requires CUDA."   
                    "Gradients will not be synchronised properly!!"           
                )
                
            self.use_ddp = False
            self.ddp_model = self.model
            
            
        
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        
        if resume_checkpoint:
            self.resume_epoch = parse_resume_epoch_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.info(f"Resuming from checkpoint {resume_checkpoint}")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        
        dist_util.sync_params(self.model.parameters())
        
    # load ema checkpoints
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)     
        
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint       
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_epoch, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.info(f"Loading EMA parameters from {ema_checkpoint}")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict) 
        dist_util.sync_params(ema_params)
        return ema_params
    
        
    # load optimizer state
    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.json(
            bf.dirname(main_checkpoint), f"opt{self.resume_epoch:06}.pt"
        )
        
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
    
    # degradation model
    def _degradation_model(self, batch_y, blur_kernel_op, alpha = 0.01):
        if self.problem_type == "deblur":
            batch_sol_Tk = deb_batch_data_solution(batch_y, blur_kernel_op, alpha = alpha)
            return batch_sol_Tk
        elif self.problem_type == "inp":
            raise NotImplementedError 
        elif self.problem_type == "sr":
            raise NotImplementedError
           
    # Training loop
    def run_training_loop(self, epochs):
        """_summary_
        """
        for epoch in range(epochs):
            # training mode
            self.ddp_model.train()
            
            self.epoch = epoch
            if (not self.lr_anneal_epochs or self.epoch + self.resume_epoch < self.lr_anneal_epochs):
                for ii, (batch_data, batch_y, blur_kernel_op) in enumerate(self.dataloader):
                    # Tikhonov data solution
                    if self.learn_alpha:
                        batch_sol_Tk = self._degradation_model(batch_y, blur_kernel_op, alpha = self.ddp_model.alpha)
                    else:
                        batch_sol_Tk = self._degradation_model(batch_y, blur_kernel_op)
                        
                    # epoch step through the model
                    self.run_epoch(batch_data, batch_sol_Tk, batch_y, blur_kernel_op)
                    
                    if self.epoch % self.log_interval == 0:
                        logger.dumpkvs()
                        
                    if self.epoch % self.save_interval == 0:
                        self.save()
                        # Run for a finite amount of time in integration tests.
                        if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.epoch > int(epochs*0.2):
                            return
                        
                # Save the last checkpoint if it wasn't already saved.
                if (self.epoch - 1) % self.save_interval != 0:
                    self.save()
            
    # epoch through the model       
    def run_epoch(self, batch_data, batch_sol_Tk, batch_y, blur_kernel_op):
        self.run_forward_backward(batch_data, batch_sol_Tk, batch_y, blur_kernel_op)
        took_epoch = self.mp_trainer.optimize(self.opt)
        
        if took_epoch:
            self._update_ema()
            
        self._anneal_lr()
        self.log_epoch()
    
    # forward and backward function
    def run_forward_backward(self, batch_data, batch_sol_Tk, batch_y, blur_kernel_op):
        self.mp_trainer.zero_grad() 
        t,weights = self.schedule_sampler.sample(batch_data.shape[0], dist_util.dev())
        
        # - compute the losses
        compute_losses = functools.partial(
                    self.diffusion.training_losses,   
                    self.ddp_model,
                    batch_data,
                    t,
                    batch_sol_Tk,
                ) 
        losses_x_pred = compute_losses()
        x_pred = losses_x_pred["pred_xstart"]
        
        if self.consistency_loss:   
            losses_x_cons = self.ddp_model.consistency_loss(x_pred, batch_y, blur_kernel_op)
        
        loss = losses_x_cons
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses_x_pred["loss"].detach()
            )
            
        loss += (losses_x_pred["loss"]*weights).mean()
        log_loss_dict(
            self.diffusion, t, {k:v*weights for k,v in losses_x_pred.items()}
        )
        self.mp_trainer.backward(loss)
    
    # - update the ema model
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate = rate)
    
    # - anneal the learning rate   
    def _anneal_lr(self):
        if not self.lr_anneal_epochs:
            return
        else:
            frac_done = (self.epoch + self.resume_epoch) / self.lr_anneal_epochs
            lr = self.lr * max(1e-20, 1.0 - frac_done)
            
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr

    # - log the epoch
    def log_epoch(self):
        logger.logkv("epoch", self.epoch + self.resume_epoch)
        logger.logkv("samples", (self.epoch + self.resume_epoch + 1) * self.global_batch)
    
    # - save the checkpoint model   
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"Saving EMA checkpoint for rate {rate}...")
                if not rate:
                    filename = f"model_{(self.epoch+self.resume_epoch):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.epoch+self.resume_epoch):06d}.pt"
                with bf.BlobFile(bf.join(self.save_checkpoint_dir, filename), "wb") as f:
                    torch.save(state_dict, f)
                    
        save_checkpoint(0, self.mp_trainer.master_params)
        
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)
            
        if dist.get_rank() ==0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.epoch+self.resume_epoch):06d}.pt"), "wb",
                ) as f:
                    torch.save(self.opt.state_dict(), f)
                    
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
def find_ema_checkpoint(main_checkpoint, epoch, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(epoch):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
