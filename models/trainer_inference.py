import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from utils.utils import inverse_image_transform, image_transform


class backward_inference:
    def __init__(self, scheduler, save_path, save_progressive):
        self.scheduler = scheduler
        self.save_path = save_path
        self.save_progressive = save_progressive
            
    def reverse_process(self, model:nn.Module, cond:torch.tensor, img_name:str, iter_num:int = 100) -> torch.tensor:
        # model 
        self.model = model
        
        # timesteps
        seq = np.sqrt(np.linspace(0, self.scheduler.num_timesteps**2, iter_num))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1
        progress_seq = seq[::max(len(seq)//10,1)]
        if progress_seq[-1] != seq[-1]:
            progress_seq.append(seq[-1])
        seq = seq[::-1]
        
        # initialisation
        x = torch.randn_like(cond)
        progress_img = []
        
        # reverse process
        for ii in range(len(seq)):
            t_i = seq[ii]
            # eps
            noise_pred = self.model(x, t_i, cond)
            x0 = self.scheduler._predict_xstart_from_eps(x, t_i, noise_pred)
            # x_t
            if ii != 0:
                t_im1 = seq[ii-1]
                x = self.scheduler.q_sample(self, x0, t_im1)
            else:
                x = x0
            
            # transforming pixels from [-1,1] ---> [0,1]    
            x_0 = inverse_image_transform(x)
        
            # save the process
            if self.save_progressive and (seq[ii] in progress_seq):
                x_show = x_0.clone().detach().cpu().numpy()       #[0,1]
                x_show = np.squeeze(x_show)
                if x_show.ndim == 3:
                    x_show = np.transpose(x_show, (1, 2, 0))
                progress_img.append(x_show)
                
                img_total = cv2.hconcat(progress_img)
                
                imsave(img_total*255., os.path.join(self.save_path, img_name))
                                        
        return x_0
                
        
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)