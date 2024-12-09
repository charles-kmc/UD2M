import torch 
import cv2

from .runners_utils import DiffusionSolver
import unfolded_models as ums
import utils as utils

class Unconditional_sampler:
    def __init__(
        self, 
        diffusion:ums.DiffusionScheduler,
        model,
        device,
        args,
        solver_type = "ddim",
    ) -> None:
        
        self.diffusion = diffusion
        self.device = device
        self.args = args
        self.solver_type = solver_type
        self.model = model
        
    def sampler(
        self, 
        im_size:torch.Tensor,
        num_timesteps:int = 100, 
        eta:float=1, 
        zeta:float=1,
    ):
        with torch.no_grad():
            # sequence of timesteps
            seq, progress_seq = self.diffusion.get_seq_progress_seq(num_timesteps)
            diffusionsolver = DiffusionSolver(seq, self.diffusion)
            
            # initilisation
            x = torch.randn(im_size).to(self.device)
            progress_img = []
            
            # reverse process
            for ii in range(len(seq)):
                t_i = seq[ii]
                print(t_i)
                out_val = self.sampling_prior(x, torch.tensor([t_i]).to(self.device))
                x0 = utils.image_transform(out_val["xstart_pred"])
                
                # sample from the predicted noise
                if seq[ii] != seq[-1] and ii < len(seq)-1: 
                    if self.solver_type == "ddim":
                        x = diffusionsolver.ddim_solver(x0, out_val["eps_pred"], ii, eta, zeta)
                    elif self.solver_type == "ddpm":
                        x = diffusionsolver.ddpm_solver(x, out_val["eps_pred"], ii, ddpm_param = self.args.ddpm_param)
                    else:
                        raise ValueError(f"This solver {self.solver_type} is not implemented. Please verify your solver.")
                else:
                    x = x0
                # save the process
                if self.args.save_progressive and (seq[ii] in progress_seq):
                    x_show = utils.get_rgb_from_tensor(utils.inverse_image_transform(x))      #[0,1]
                    progress_img.append(x_show)
                    utils.delete([x_show])
            
            if self.args.save_progressive:   
                img_total = cv2.hconcat(progress_img)
        out_val["progress_img"] = img_total
        return out_val
    
    def sampling_prior(self, xt, t):
        B,C = xt.shape[:2]
        pred_eps = self.model(xt, t)
        pred_eps, _ = torch.split(pred_eps, C, dim=1) if pred_eps.shape == (B, C * 2, *xt.shape[2:]) else (pred_eps, pred_eps)
        
        # estimate of x_start
        x = self.diffusion.predict_xstart_from_eps(xt, t, pred_eps)
        x_pred = utils.inverse_image_transform(x)
        x0 = torch.clamp(x_pred, 0.0, 1.0)
    
        return {"xstart_pred":x0, "aux":x0, "eps_pred":pred_eps}
