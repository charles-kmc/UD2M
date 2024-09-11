import torch 
from deepinv import Trainer
from deepinv.utils import plot, rescale_img
from pathlib import Path
import os
import numpy as np
import wandb
from tqdm import tqdm
import torchvision

class EarlyStopping(torch.nn.Module):
    def __init__(self, patience = 2, loss="TotalLoss", minimize = True):
        super().__init__()
        self.patience = patience  # Stop after N_stop worse steps
        self.loss = loss  # Use train or validation loss
        self.best_loss = None
        self.minimize = minimize  # +ve for min loss and -ve for max loss
        self.counter = 0
        self.stop_now = False

    def step(self, logs):
        try:
            loss = logs[self.loss]
        except:
            print(f"WARNING: Loss {self.loss} is not provided in logs {logs.keys()}", start = f"\n{100*'#'}\n and Earlystopping will be turned off", end = f"\n{100*'#'}\n")
            self.step = lambda logs: 0
        if self.best_loss is None:
            self.best_loss = loss
        elif (self.minimize and loss > self.best_loss) or ((not self.minimize) and loss < self.best_loss):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_now = True
        else: 
            self.best_loss = loss 
            self.counter = 0
    
    def reset(self):
        self.counter = 0 
        self.best_loss = None
        self.stop_now = False

## Modified to plot images with checkpoints
class TrainerPlt(Trainer):
    def __init__(self, model, pretrain_optimizer = None, save_every_n_iter = None, training_metrics=None, resume_last=False, EarlyStopper=None, **kwargs):
        super().__init__(model, **kwargs)
        if save_every_n_iter is None:
            self.save_every_n_iter = len(self.train_dataloader[0])
        else: 
            self.save_every_n_iter = int(save_every_n_iter)
        self.training_metrics = training_metrics
        self.resume_last=resume_last
        self._has_setup_train=False
        self.EarlyStopper = EarlyStopper
        

    def plot(self, epoch, physics, x, y, x_net, train=True):
        r"""
        Plot the images.

        It plots the images at the end of each epoch.

        :param int epoch: Current epoch.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Network reconstruction.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        """
        post_str = "Training" if train else "Eval"
        if self.plot_images and ((epoch + 1) % self.freq_plot == 0):
            imgs, titles, grid_image, caption = self.prepare_images(
                physics, x, y, x_net
            )

            plot(
                imgs,
                titles=titles,
                show=self.plot_images,
                return_fig=True,
                rescale_mode="clip",
                save_dir = os.path.join(
                    Path(self.save_path), Path(post_str + "_im_{}".format(epoch))
                )
            )

            if self.wandb_vis:
                log_dict_post_epoch = {}
                if hasattr(physics, "filter"):
                    kernel_im = wandb.Image(physics.filter.clone().cpu().squeeze(dim=0))
                    log_dict_post_epoch["Kernel"] = kernel_im
                images = wandb.Image(
                    grid_image,
                    caption=caption,
                )
                log_dict_post_epoch[post_str + " samples"] = images
                log_dict_post_epoch["step"] = epoch
                wandb.log(log_dict_post_epoch)
    
    def compute_metrics(self, x, x_net, y, physics, logs, train=True):
        r"""
        Compute the metrics.

        It computes the metrics over the batch.

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param dict logs: Dictionary containing the logs for printing the training progress.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :returns: The logs with the metrics.
        """
        # Compute the metrics over the batch
        with torch.no_grad():
            thisiterlog = {}
            for k, l in enumerate(self.training_metrics if train else self.metrics):
                metric = l(x=x, x_net=x_net, y=y, physics=physics)

                current_log = (
                    self.logs_metrics_train[k] if train else self.logs_metrics_eval[k]
                )
                current_log.update(metric.detach().cpu().numpy())
                logs[l.__class__.__name__] = current_log.avg
                thisiterlog[l.__class__.__name__] = metric.detach().cpu().numpy()

        return logs, thisiterlog
    
    def step(self, epoch, progress_bar, train=True, last_batch=False):
        r"""
        Train/Eval a batch.

        It performs the forward pass, the backward pass, and the evaluation at each iteration.

        :param int epoch: Current epoch.
        :param tqdm progress_bar: Progress bar.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param bool last_batch: If ``True``, the last batch of the epoch is being processed.
        :returns: The current physics operator, the ground truth, the measurement, and the network reconstruction.
        """

        # random permulation of the dataloaders
        G_perm = np.random.permutation(self.G)

        for g in G_perm:  # for each dataloader
            x, y, physics_cur = self.get_samples(self.current_iterators, g)
            # Compute loss and perform backprop
            x_net, logs, current_loss = self.compute_loss(physics_cur, x, y, train=train)
            # Log metrics
            logs, current_metrics = self.compute_metrics(x, x_net, y, physics_cur, logs, train=train)
            current_losses = {**current_loss, **current_metrics}
            
            # Update the progress bar
            progress_bar.set_postfix(logs)

        if (train and self.global_iter % self.save_every_n_iter == 0 and self.global_iter >0) or (not train and last_batch):
            if not train:
                logs["epoch"] = epoch
            if train and self.EarlyStopper:  # Determine early stopping
                self.EarlyStopper.step(logs)
            self.store_logs(logs, train, physics_cur)  # Log metrics to wandb
        if last_batch:
            if self.verbose and not self.show_progress_bar:
                if self.verbose_individual_losses:
                    print(
                        f"{'Train' if train else 'Eval'} epoch {epoch}:"
                        f" {', '.join([f'{k}={round(v, 3)}' for (k, v) in logs.items()])}"
                    )
                else:
                    print(
                        f"{'Train' if train else 'Eval'} epoch {epoch}: Total loss: {logs['TotalLoss']}"
                    )
            logs["step"] = epoch
            self.plot(epoch, physics_cur, x, y, x_net, train=train)  # plot images
        self.global_iter += int(train)
    
    def setup_train(self):
        if self.resume_last:
            import datetime
            # Get path of last saved model
            d = sorted(
                os.listdir(self.save_path),
                key = lambda x: datetime.datetime.strptime(x, '%y-%m-%d-%H:%M:%S')
            )[-1]
            d = os.path.join(self.save_path, d)
            ckpts = []
            for s in os.listdir(d):
                if 'ckp_' in s:
                    ckpts.append(s)
            if len(ckpts)>1:
                self.ckpt_pretrained = sorted(
                    ckpts,
                    key = lambda x: int(x.partition('.pth')[0][4:])
                )[-1]
                self.ckpt_pretrained = os.path.join(d, self.ckpt_pretrained)
        save_path_old=self.save_path
        super().setup_train()
        self.save_path = save_path_old
        self.global_iter = self.epoch_start*(
                len(self.train_dataloader[self.G - 1]) - 
                int(self.train_dataloader[self.G - 1].drop_last
            ))
        self.training_metrics = self.metrics if self.training_metrics is None else self.training_metrics
        if not isinstance(self.metrics, list) or isinstance(self.metrics, tuple):
            self.training_metrics = [self.training_metrics]
        self._has_setup_train = True 

    def store_logs(self, logs, train, physics):
        if self.wandb_vis:
            with torch.no_grad():
                tr = 'train' if train else 'eval'
                new_logs = {}
                for k, v in logs.items():
                    new_logs[tr + '/' +  k] = v
                if train:
                    new_logs = {**new_logs,  "tr_iter":self.global_iter+1}
                    if hasattr(self.model, 'log_param_dict'):
                        new_logs = {**new_logs, **self.model.log_param_dict(physics)}
                wandb.log(new_logs)
                
        self.reset_metrics(train=train)
    
    def prepare_images(self, physics_cur, x, y, x_net):
        r"""
        Prepare the images for plotting.

        It prepares the images for plotting by rescaling them and concatenating them in a grid.

        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Reconstruction network output.
        :returns: The images, the titles, the grid image, and the caption.
        """
        with torch.no_grad():
            if (
                self.plot_measurements
                and len(y.shape) == len(x.shape)
                and y.shape != x.shape
            ):
                y_reshaped = torch.nn.functional.interpolate(y, size=x.shape[2])
                if hasattr(physics_cur, "A_adjoint"):
                    imgs = [y_reshaped, physics_cur.A_adjoint(y), x_net, x]
                    caption = (
                        "From top to bottom: input, backprojection, output, target"
                    )
                    titles = ["Input", "Backprojection", "Output", "Target"]
                else:
                    imgs = [y_reshaped, x_net, x]
                    titles = ["Input", "Output", "Target"]
                    caption = "From top to bottom: input, output, target"
            else:
                if hasattr(physics_cur, "A_adjoint"):
                    if isinstance(physics_cur, torch.nn.DataParallel):
                        back = physics_cur.module.A_adjoint(y)
                    else:
                        back = physics_cur.A_adjoint(y)
                    
                    imgs = [y, x_net, x]
                    titles = ["Observation", "Output", "Target"]
                    caption = "From top to bottom: observation, output, target"
                else:
                    imgs = [x_net, x]
                    caption = "From top to bottom: output, target"
                    titles = ["Output", "Target"]
                
                

            vis_array = torch.cat(imgs, dim=0)
            for i in range(len(vis_array)):
                vis_array[i] = rescale_img(vis_array[i], rescale_mode="min_max")
            if hasattr(physics_cur, "filter"):
                        s = physics_cur.filter.shape
                        fil = physics_cur.filter 
                        if fil.shape[0] == 1:
                            fil = fil.expand(y.shape[0],-1,-1,-1)
                        for i in range(len(fil)):
                            fil[i] = rescale_img(fil[i], rescale_mode="min_max")
                        if s[-1] < y.shape[-1]/2:
                            vis_array[:y.shape[0],:,:s[2],:s[3]] = fil
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])

        return imgs, titles, grid_image, caption
    
    def compute_loss(self, physics, x, y, train=True):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        self.optimizer.zero_grad()

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics)

        if train or self.display_losses_eval:
            # Compute the losses
            loss_total = 0
            for k, l in enumerate(self.losses):
                loss = l(x=x, x_net=x_net, y=y, physics=physics, model=self.model)
                loss_total += loss.mean()
                if len(self.losses) > 1 and self.verbose_individual_losses:
                    current_log = (
                        self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                    )
                    current_log.update(loss.detach().cpu().numpy())
                    cur_loss = current_log.val
                    logs[l.__class__.__name__] = cur_loss

            current_log = (
                self.logs_total_loss_train if train else self.logs_total_loss_eval
            )
            current_log.update(loss_total.item())
            logs[f"TotalLoss"] = current_log.avg

        if train:
            loss_total.backward()  # Backward the total loss

            norm = self.check_clip_grad()  # Optional gradient clipping
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            # Optimizer step
            self.optimizer.step()
            logs["learning_rate"] = self.optimizer.param_groups[0]['lr']

        return x_net, logs, {f"TotalLoss":loss_total if train else None}
    
    def reset_metrics(self, train = True):
        if train:
            for l in self.logs_metrics_train:
                l.reset()
            for l in self.logs_losses_train:
                l.reset()
            self.logs_total_loss_train.reset()
        else: 
            for l in self.logs_metrics_eval:
                l.reset()
            for l in self.logs_losses_eval:
                l.reset()
            self.logs_total_loss_eval.reset()

    def train(
        self,
        epochs = None
    ):
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        :returns: The trained model.
        """
        if not self._has_setup_train:
            self.setup_train()
            self.evaluation(0)  # Evaluate model in initial state
        else: 
            self.epoch_start = self.epochs 
            self.epochs += epochs if epochs else 1 

        for epoch in range(self.epoch_start, self.epochs):
            ## Training
            self.current_iterators = [iter(loader) for loader in self.train_dataloader]
            batches = len(self.train_dataloader[self.G - 1]) - int(
                self.train_dataloader[self.G - 1].drop_last
            )

            self.model.train()
            for i in (
                progress_bar := tqdm(
                    range(batches),
                    ncols=150,
                    disable=(not self.verbose or not self.show_progress_bar),
                )
            ):
                progress_bar.set_description(f"Train epoch {epoch + 1}")
                self.step(
                    epoch, progress_bar, train=True, last_batch=(i == batches - 1)
                )
                if self.EarlyStopper:
                    if self.EarlyStopper.stop_now:
                        break

            self.loss_history.append(self.logs_total_loss_train.avg)

            if self.scheduler:
                self.scheduler.step()

            ## Evaluation 
            self.evaluation(epoch + 1)  # evaluate trained model

            # Saving the model
            self.save_model(epoch, self.eval_metrics_history)

        if self.wandb_vis:
            wandb.save("model.h5")

        return self.model

    def evaluation(self, epoch):
        with torch.no_grad():
            self.current_iterators = [
                iter(loader) for loader in self.eval_dataloader
            ]
            batches = len(self.eval_dataloader[self.G - 1]) - int(
                self.eval_dataloader[self.G - 1].drop_last
            )

            # self.model.eval()
            for i in (
                progress_bar := tqdm(
                    range(batches),
                    ncols=150,
                    disable=(not self.verbose or not self.show_progress_bar),
                )
            ):
                progress_bar.set_description(f"Eval epoch {epoch}")
                self.step(
                    epoch, progress_bar, train=False, last_batch=(i == batches - 1)
                )

            for l in self.logs_losses_eval:
                self.eval_metrics_history[l.name] = l.avg
    
    
    
    def save_model(self, epoch, eval_metrics=None):
        r"""
        Save the model.

        It saves the model every ``ckp_interval`` epochs.

        :param int epoch: Current epoch.
        :param None, float eval_metrics: Evaluation metrics across epochs.
        """
        if (epoch > 0 and epoch % self.ckp_interval == 0) or epoch + 1 == self.epochs:
            os.makedirs(str(self.save_path), exist_ok=True)
            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "loss": self.loss_history,
                "optimizer": self.optimizer.state_dict(),
            }
            if eval_metrics is not None:
                state["eval_metrics"] = eval_metrics
            torch.save(
                state,
                os.path.join(
                    Path(self.save_path), Path("ckp.pth.tar")
                ),
            )

class UnfoldedTrainer(TrainerPlt):
    ### Base trainer for unfolded class which permits expanding size of model with greedy learning
    def train(self, epochs=None, no_expand_levels = 0):
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        :returns: The trained model.
        """
        if not self._has_setup_train:
            self.setup_train()
            self.evaluation(0)  # Evaluate model in initial state
        else: 
            self.epoch_start = self.epochs 
            self.epochs += epochs if epochs else 1 
        save_path_base = self.save_path
        epochs_done = 0
        tr_mode = "trainall"
        for l in range(3*no_expand_levels+1):
            if l > 0:  # Append new level to model
                if tr_mode == "trainall":
                    print(f"Expanding model to {(l+1)//3+1} layers")
                    new_params = self.model.add_new_level()
                    ## Freeze previous Layers
                    set_optimizer_grad(self.optimizer, False, train_prior=False)
                    self.optimizer.add_param_group({"params":new_params})
                    tr_mode = "trainlast"
                    print(f"Training Layer {(l+1)//3+1} only")
                elif tr_mode == "trainlast":
                    set_optimizer_grad(self.optimizer, True, train_prior=False)
                    print(f"Training All {(l+1)//3+1} layers")
                    tr_mode = "trainallmodel"
                elif tr_mode == "trainallmodel":
                    set_optimizer_grad(self.optimizer, True, train_prior=True)
                    print(f"Training All {(l+1)//3+1} layers and prior")
                    tr_mode = "trainall"
            if self.EarlyStopper:
                self.EarlyStopper.reset()
            self.save_path = save_path_base + str(int(self.model.max_iter))
            for epoch in range(epochs_done, epochs_done + self.epochs):
                ## Training
                self.current_iterators = [iter(loader) for loader in self.train_dataloader]
                batches = len(self.train_dataloader[self.G - 1]) - int(
                    self.train_dataloader[self.G - 1].drop_last
                )

                self.model.train()
                for i in (
                    progress_bar := tqdm(
                        range(batches),
                        ncols=150,
                        disable=(not self.verbose or not self.show_progress_bar),
                    )
                ):
                    progress_bar.set_description(f"Train epoch {epoch + 1}")
                    self.step(
                        epoch, progress_bar, train=True, last_batch=(i == batches - 1)
                    )
                    if self.EarlyStopper:
                        if self.EarlyStopper.stop_now:
                            break

                self.loss_history.append(self.logs_total_loss_train.avg)

                # if self.scheduler:
                #     self.scheduler.step()

                ## Evaluation 
                self.evaluation(epoch + 1)  # evaluate trained model

                # Saving the model
                self.save_model(epoch, self.eval_metrics_history)
                epochs_done += 1

                if self.EarlyStopper:
                        if self.EarlyStopper.stop_now:
                            break
            

        if self.wandb_vis:
            wandb.save("model.h5")

        return self.model

class DiffusionTrainer(TrainerPlt):
    def prepare_images(self, physics_cur, x, y, x_net):
        r"""
        Prepare the images for plotting.

        It prepares the images for plotting by rescaling them and concatenating them in a grid.

        :param deepinv.physics.Physics physics_cur: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param torch.Tensor x_net: Reconstruction network output.
        :returns: The images, the titles, the grid image, and the caption.
        """
        with torch.no_grad():
            y, xt = y
            imgs = [y, xt, x_net, x]
            titles = ["y", "xt", "Output", "Target"]
            caption = f"From top to bottom: y, xt, output, target with t = {physics_cur.diff_physics.ts.clone().cpu().flatten().numpy()} from left to right."
            vis_array = torch.cat(imgs, dim=0)
            for i in range(len(vis_array)):
                vis_array[i] = rescale_img(vis_array[i], rescale_mode="min_max")
            if hasattr(physics_cur, "filter"):
                        s = physics_cur.filter.shape
                        fil = physics_cur.filter 
                        if fil.shape[0] == 1:
                            fil = fil.expand(y.shape[0],-1,-1,-1)
                        for i in range(len(fil)):
                            fil[i] = rescale_img(fil[i], rescale_mode="min_max")
                        if s[-1] < y.shape[-1]/2:
                            vis_array[:y.shape[0],:,:s[2],:s[3]] = fil
            grid_image = torchvision.utils.make_grid(vis_array, nrow=y.shape[0])
        return imgs, titles, grid_image, caption


def set_optimizer_grad(optimizer, requires_grad, train_prior = True):
    for pg in optimizer.param_groups:
        if hasattr(pg, "tag") and pg["tag"]=="prior" and not train_prior:
            continue
        for p in pg["params"]:
            p.requires_grad = requires_grad
