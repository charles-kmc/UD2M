import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_deblur import MotionBlurOperator, GaussialBlurOperator
from scipy import ndimage

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from utils.utils import Metrics, DotDict, inverse_image_transform, image_transform
from models.model import FineTuningDiffusionModel
from models.pipeline import Pipeline
from DPIR.dpir_models import DPIR_deb, dpir_solver


class DiffPIR:
    def __init__(self, model_freeze_name, device, args, cond_used = True, epoch = 5, model_used = None):
        self.args = args
        self.device = device
        self.date  = datetime.datetime.now().strftime("%d-%m-%Y")
        self.epoch = epoch
        self.model_used = model_used
        self.model_name = f"{args.task}_ema_model_rate_0.99_{epoch:03d}"
        self.model_freeze_name = model_freeze_name
        self.cond_used = cond_used
        
        # date
        self.date  = datetime.datetime.now().strftime("%d-%m-%Y")
        
        # noise scheduler
        self._noise_scheduler()
        
        # image paths
        self.images_paths = util.get_image_paths(self.args.test_dataset_dir)
        
        # load model and diffusion
        self._load_models(self.model_name, self.model_freeze_name)
        
        # metrices
        self.metric_module = Metrics(self.device)
        
    def run(self, noise_val, blur_mode, lambda_ = 7, zeta = 0.3):
        self.lambda_ = lambda_
        self.zeta = zeta
        self.noise_val = noise_val
        self.sigma = max(self.noise_val, 0.001)
        self.blur_mode = blur_mode
        
        # DPIR module
        denoiser_name = "drunet_color"   
        self._load_conditioning_model(denoiser_name)

        # make dirs and get loggers
        self._create_dir()
        self._start_logger()

        # get t start  
        noise_model_t = utils_model.find_nearest(self.reduced_alpha_cumprod, 2 * noise_val)
        noise_model_t = 0
        noise_inti_img = 50 / 255
        t_start = utils_model.find_nearest(self.reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
        t_start = self.args.num_train_timesteps - 1              

        # monitor results
        test_results = DotDict()
        test_results.psnr = []
        test_results.ssim = []
        test_results.lpips = []
            
        # get image
        for idx, img in enumerate(self.images_paths):
            # get kernel            
            k = self._get_kernel(idx)
            k_4d = torch.from_numpy(k).to(self.device)
            k_4d = torch.einsum('ab,cd->abcd',torch.eye(3).to(self.device),k_4d)
            
            # get high resolution image 
            img_name, _ = os.path.splitext(os.path.basename(img))
            ref_im = util.imread_uint(img, n_channels=self.args.n_channels)
            ref_im = util.modcrop(ref_im, 8)
            
            # mode='wrap' is important for analytical solution
            img_L = ndimage.convolve(ref_im, np.expand_dims(k, axis=2), mode='wrap')
            img_L = util.uint2single(img_L)
            
            # low resolution image
            np.random.seed(seed=0)  
            img_L = img_L * 2 - 1
            img_L += np.random.normal(0, self.noise_val * 2, img_L.shape) # add AWGN
            img_L = img_L / 2 + 0.5
            
            # get rhos and sigmas
            self.rhos, self.sigmas, self.sigma_ks = self._rhos_sigmas()
            
            # pre-calculation
            y = util.single2tensor4(img_L).to(self.device)   #(1,3,256,256)
            k_tensor = util.single2tensor4(np.expand_dims(k, 2)).to(self.device)
            FB, FBC, F2B, FBFy = sr.pre_calculate(y, k_tensor, self.args.sf)
            
            # conditioning
            cond = dpir_solver(y, k_tensor.squeeze(), self.dpir, iter_num = 2)
            self.model  = lambda x_t, t: self.model(x_t, t, cond)
            
            # Initialisation
            t_y = utils_model.find_nearest(self.reduced_alpha_cumprod, 2 * self.noise_val)
            sqrt_alpha_effective = self.sqrt_alphas_cumprod[t_start] / self.sqrt_alphas_cumprod[t_y]
            x = sqrt_alpha_effective * (image_transform(y)) + torch.sqrt(self.sqrt_1m_alphas_cumprod[t_start]**2 - \
                    sqrt_alpha_effective**2 * self.sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)
            
            
            # main iterations
            progress_img = []
            self._get_sequence_timesteps()
                    
            # reverse diffusion for one image from random noise
            for ii in range(len(self.seq)):
                curr_sigma = self.sigmas[self.seq[ii]].cpu().numpy()
                # time step associated with the noise level sigmas[i]
                t_i = utils_model.find_nearest(self.reduced_alpha_cumprod,curr_sigma)
                # skip iters
                if t_i > t_start:
                    continue
                for u in range(self.args.iter_num_U):
                    
                    # step 1, reverse diffsuion step
                    x0 = utils_model.model_fn(
                                            x, 
                                            noise_level=curr_sigma*255, 
                                            model_out_type="pred_x_prev",
                                            model_diffusion = self.model, 
                                            diffusion = self.diffusion, 
                                            ddim_sample = self.args.ddim_sample, 
                                            alphas_cumprod=self.alphas_cumprod
                                        )
                    
                    # step 2, FFT
                    if self.seq[ii] != self.seq[-1]:
                        if self.args.sub_1_analytic:
                            tau = self.rhos[t_i].float().repeat(1, 1, 1, 1)
                            # when noise level less than given image noise, skip
                            if ii < self.args.num_train_timesteps-noise_model_t: 
                                x0_p = x0 / 2 + 0.5
                                x0_p = sr.data_solution(x0_p.float(), FB, FBC, F2B, FBFy, tau, self.args.sf)
                                x0_p = x0_p * 2 - 1
                                # effective x0
                                x0 = x0 + self.args.guidance_scale * (x0_p-x0)
                            else:
                                x0 = utils_model.model_fn(
                                                        x, 
                                                        noise_level=curr_sigma*255, 
                                                        model_out_type="pred_x_prev",
                                                        model_diffusion=self.model, 
                                                        diffusion=self.diffusion, 
                                                        ddim_sample=self.args.ddim_sample, 
                                                        alphas_cumprod=self.alphas_cumprod
                                                    )
                                
                        else:
                            # zeta=0.28; lambda_=7
                            x0 = x0.requires_grad_()
                            # first order solver
                            def Tx(x):
                                x = x / 2 + 0.5
                                pad_2d = torch.nn.ReflectionPad2d(k.shape[0]//2)
                                x_deblur = F.conv2d(pad_2d(x), k_4d)
                                return x_deblur
                            
                            norm_grad, norm = utils_model.grad_and_value(
                                                                        operator=Tx,
                                                                        x=x0, 
                                                                        x_hat=x0, 
                                                                        measurement=y
                                                                    )
                            x0 = x0 - norm_grad * norm / (self.rhos[t_i])
                            x0 = x0.detach_()
                            pass                               
                    
                    if not (self.seq[ii] == self.seq[-1] and u == self.args.iter_num_U-1):
                        #x = sqrt_alphas_cumprod[t_i] * (x0) + (sqrt_1m_alphas_cumprod[t_i]) *  torch.randn_like(x)
                        
                        t_im1 = utils_model.find_nearest(self.reduced_alpha_cumprod, self.sigmas[self.seq[ii+1]].cpu().numpy())
                        # calculate \hat{\eposilon}
                        eps = (x - self.sqrt_alphas_cumprod[t_i] * x0) / self.sqrt_1m_alphas_cumprod[t_i]
                        
                        eta_sigma = self.args.eta * self.sqrt_1m_alphas_cumprod[t_im1] / self.sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(self.betas[t_i])
                        
                        x = self.sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(self.sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                    + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * self.sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                    else:
                        pass
                        
                    # set back to x_t from x_{t-1}
                    if u < self.args.iter_num_U-1 and self.seq[ii] != self.seq[-1]:
                        sqrt_alpha_effective = self.sqrt_alphas_cumprod[t_i] / self.sqrt_alphas_cumprod[t_im1]
                        
                        x = sqrt_alpha_effective * x + torch.sqrt(self.sqrt_1m_alphas_cumprod[t_i]**2 - \
                                sqrt_alpha_effective**2 * self.sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)

                # save the process
                x_0 = inverse_image_transform(x)
                if self.args.save_progressive and (self.seq[ii] in self.progress_seq):
                    x_show = x_0.clone().detach().cpu().numpy()       #[0,1]
                    x_show = np.squeeze(x_show)
                    if x_show.ndim == 3:
                        x_show = np.transpose(x_show, (1, 2, 0))
                    progress_img.append(x_show)
                    if self.args.log_process:
                        self.logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(self.seq[ii], t_i, np.max(x_show), np.min(x_show)))
                        
                x0_uint = util.tensor2uint(x_0)
                ref_tensor = torch.from_numpy(np.transpose(ref_im, (2, 0, 1)))[None,:,:,:].to(self.device)
                ref_tensor = ref_tensor / 255.
                psnr = self.metric_module.psnr_function(x_0, ref_tensor).numpy()
                ssim = self.metric_module.psnr_ssim(x_0, ref_tensor).numpy()
                lpips_score = self.metric_module.lpips_function(x_0, ref_tensor).detach().cpu().squeeze().numpy()
                
                test_results.lpips.append(lpips_score)
                test_results.psnr.append(psnr)
                test_results.ssim.append(ssim)
                self.logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB LPIPS: {:.4f}'.format(idx+1, img_name, psnr, lpips_score))

                if self.args.save_im:
                    img_L = util.single2uint(img_L)
                    util.imsave(x0_uint, os.path.join(self.results_dir_recons, img_name+'.png'))
                    util.imsave(ref_im, os.path.join(self.results_dir_recons, img_name+'.png'))
                    util.imsave(img_L, os.path.join(self.results_dir_deg, img_name+'.png'))
                    
                    k_v = k/np.max(k)*1.0
                    k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                    k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I = cv2.resize(img_L, (self.args.sf*img_L.shape[1], self.args.sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                    img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                    util.imshow(np.concatenate([img_I, x0_uint, ref_im], axis=1), title='LR / Recovered / Ground-truth')
                    util.imsave(np.concatenate([img_I, x0_uint, ref_im], axis=1), os.path.join(self.results_dir_leh, img_name+'.png')) if self.arg.save_LEH else None

                if self.args.save_progressive:
                    img_total = cv2.hconcat(progress_img)
                    util.imsave(img_total*255., os.path.join(self.results_dir_progr, img_name+'.png'))
                
        # Average PSNR and LPIPS
        ave_psnr = sum(test_results.psnr) / len(test_results.psnr)
        ave_ssim = sum(test_results.ssim) / len(test_results.ssim)
        ave_lpips = sum(test_results.lpips) / len(test_results.lpips)
        self.logger.info('------> Average LPIPS - sigma: ({:.3f}): {:.4f}'.format(noise_val, ave_lpips))
        self.logger.info('------> Average PSNR - sigma: ({:.3f}): {:.4f} dB'.format(noise_val, ave_psnr))
        self.logger.info('------> Average SSIM - sigma: ({:.3f}): {:.4f} dB'.format(noise_val, ave_ssim))

    def _load_conditioning_model(self, denoiser_name, solver = "dpir"):
        if solver == "dpir":
            net_path = "/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo"
            self.dpir = DPIR_deb(model_name = denoiser_name, noise_val = self.noise_val, pretrained_pth = net_path)    
        else:
            raise ValueError(f"This solver: {solver} is not yet implemented !!!")
        
    def _get_sequence_timesteps(self):
        if self.args.skip_type == 'uniform':
            self.seq = [ii*self.args.skip for ii in range(self.args.iter_num)]
            if self.args.skip > 1:
                self.seq.append(self.args.num_train_timesteps-1)
        elif self.args.skip_type == "quad":
            self.seq = np.sqrt(np.linspace(0, self.args.num_train_timesteps**2, self.args.iter_num))
            self.seq = [int(s) for s in list(self.seq)]
            self.seq[-1] = self.seq[-1] - 1
        self.progress_seq = self.seq[::max(len(self.seq)//10,1)]
        if self.progress_seq[-1] != self.seq[-1]:
            self.progress_seq.append(self.seq[-1]) 
                        
    def _get_kernel(self, seed):
        np.random.seed(seed=seed*10)  # for reproducibility of blur kernel for each image
        if self.blur_mode == 'gaussian':
            kernel_std = 3.0
            kernel_std_i = kernel_std * np.abs(np.random.rand()*2+1)
            kernel = GaussialBlurOperator(kernel_size=self.args.kernel_size, intensity=kernel_std_i, device=self.device)
        elif self.blur_mode == 'motion':
            kernel_std = 0.5
            kernel = MotionBlurOperator(kernel_size=self.args.kernel_size, intensity=kernel_std, device=self.device)
        k_tensor = kernel.get_kernel().to(self.device, dtype=torch.float)
        k = k_tensor.clone().detach().cpu().numpy()       #[0,1]
        k = np.squeeze(k)
        k = np.squeeze(k)  
        return k 
        
    def _rhos_sigmas(self):
        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(self.args.num_train_timesteps):
            sigmas.append(self.reduced_alpha_cumprod[self.args.num_train_timesteps-1-i])
            sigma_ks.append((self.sqrt_1m_alphas_cumprod[i]/self.sqrt_alphas_cumprod[i]))
            rhos.append(self.lambda_*(self.sigma**2)/(sigma_ks[i]**2))    
        rhos, sigmas, sigma_ks = torch.tensor(rhos).to(self.device), torch.tensor(sigmas).to(self.device), torch.tensor(sigma_ks).to(self.device)
        
        return rhos, sigmas, sigma_ks
    
    def _noise_scheduler(self):
        # noise schedule 
        beta_start                   = 0.1 / 1000
        beta_end                     = 20 / 1000
        betas                        = np.linspace(beta_start, beta_end, self.args.num_train_timesteps, dtype=np.float32)
        self.betas                   = torch.from_numpy(betas).to(self.device)
        self.alphas                  = 1.0 - self.betas
        self.alphas_cumprod          = np.cumprod(self.alphas.cpu(), axis=0)
        self.sqrt_alphas_cumprod     = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod)
        self.reduced_alpha_cumprod   = torch.div(self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod)       
        self.noise_inti_img          = 50 / 255
        
    
    # Loading models 
    def _load_models(self, model_name, model_freeze_name):
        if ".pt" not in model_name or ".pth" not in model_name:
            raise ValueError(f"this model name {model_name} is not in a good format !!!")
        if ".pt" not in model_freeze_name or ".pth" not in model_freeze_name:
            raise ValueError(f"this model name {model_name} is not in a good format !!!")
        else:
            model_freeze_name = model_freeze_name.split(".")[0]
        # frozen model and diffusion scheduler
        frozen_model_path = os.path.join(self.args.model_path, model_freeze_name)
        frozen_model_config = dict(
            frozen_model_path=frozen_model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16"
        ) 
        frozen_model_args = utils_model.create_argparser(frozen_model_config).parse_args([])
        frozen_model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(frozen_model_args, model_and_diffusion_defaults().keys()))
        frozen_model.load_state_dict(
            dist_util.load_state_dict(frozen_model_args.frozen_model_path, map_location="cpu")
        )
        
        if self.cond_used:
            # Load new model and checkpoints
            if self.model_used is None:
                model_path = os.path.join(self.args.model_path, model_name)
                self.model = FineTuningDiffusionModel(256, 3, 128, 3, self.device, frozen_model = frozen_model)  
                self.model.load_state_dict(dist_util.load_state_dict(model_path)["model_state_dict"], map_location="cpu") 
            elif self.model_used == "simple":
                model_path = os.path.join(self.args.model_path, "simple", model_name)
                self.model = Pipeline(frozen_model, self.device)
                self.model.load_state_dict(dist_util.load_state_dict(model_path)["model_state_dict"], map_location="cpu")
            else:
                raise ValueError("This model is implemented !!!")
        else:
            # Using only the frozen model for inference
            self.model = frozen_model 
        
        # setting model for inference    
        self.model.eval()
        self._frozen_model()
        self.model = self.model.to(self.device)
        
    # freeze parameters of the model
    def _frozen_model(self):
        for _k, v in self.model.named_parameters():
            v.requires_grad = False
    
    # Loggers
    def _start_logger(self):
        self.result_name = 'logger'
        logger_name = self.result_name
        utils_logger.logger_info(logger_name, log_path=os.path.join(self.results_dir, logger_name+'.log'))
        self.logger = logging.getLogger(logger_name)
        self.logger.info('image sigma:{:.3f}'.format(self.sigma))
        self.logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(self.args.eta, self.zeta, self.lambda_, self.args.guidance_scale))
        self.logger.info('skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(self.args.skip_type, self.args.skip))

    # Important directories
    def _create_dir(self):
        if self.cond_used:
            folder_name = "CDM"
        else:
            folder_name = "non-CDM"
        self.results_dir = os.path.join(self.args.results_dir, "inference", "DiffPIR", self.date, self.args.task, self.args.testset_name, f"noise_val_{self.noise_val}" , self.args.blur_mode+f"_{self.args.kernel_size}", self.args.iter_num, folder_name)
        
        self.results_dir_deg = os.path.join(self.results_dir, "degs")
        self.results_dir_refs = os.path.join(self.results_dir, "refs")
        self.results_dir_recons = os.path.join(self.results_dir, "recons")
        self.results_dir_progr = os.path.join(self.results_dir, "progrs")
        self.results_dir_leh = os.path.join(self.results_dir, "leh")
        self.results_dir_data = os.path.join(self.results_dir, "data")
        
        os.makedirs(self.results_dir_deg, exist_ok=True) if self.arg.save_im else None
        os.makedirs(self.results_dir_refs, exist_ok=True) if self.arg.save_im else None
        os.makedirs(self.results_dir_recons, exist_ok=True) if self.arg.save_im else None
        os.makedirs(self.results_dir_progr, exist_ok=True) if self.arg.save_progressive else None
        os.makedirs(self.results_dir_data, exist_ok=True) if self.arg.save_data else None
        os.makedirs(self.results_dir_leh, exist_ok=True) if self.arg.save_LEH else None