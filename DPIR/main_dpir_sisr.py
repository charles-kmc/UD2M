import os.path
import glob
import cv2
import logging
import time

import numpy as np
import scipy.io as sio
from datetime import datetime
from collections import OrderedDict
#import hdf5storage

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils import fspecial_gaussian

from metrics import *
import lpips

"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_color         # model_name, for color images
   |--drunet_gray
|--testset                 # testsets
|--results                 # results
# --------------------------------------------


How to run:
step 1: download [drunet_gray.pth, drunet_color.pth, ircnn_gray.pth, ircnn_color.pth] from https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
step 2: set your own testset 'testset_name' and parameter setting such as 'noise_level_img', 'iter_num'. 
step 3: 'python main_dpir_sisr.py'

"""

def main(noise_level = 0, testset_name = "data"):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = noise_level    #0/255.0            # set AWGN noise level for LR image, default: 0, 
    noise_level_model = noise_level_img  # setnoise level of model, default 0
    model_name = 'ircnn_color'          # set denoiser, | 'drunet_color' | 'ircnn_gray' | 'drunet_gray' | 'ircnn_color'
    x8 = True                            # default: False, x8 to boost performance
    test_sf = [4]                        # set scale factor, default: [2, 3, 4], [2], [3], [4]
    iter_num = 24                        # set number of iterations, default: 24 for SISR
    modelSigma1 = 49                     # set sigma_1, default: 49
    classical_degradation = True         # set classical degradation or bicubic degradation

    show_img = False                     # default: False
    save_L = True                        # save LR image
    save_E = True                        # save estimated image
    save_LEH = False                     # save zoomed LR, E and H images

    # --------------------------------
    # load kernel
    # --------------------------------
    kernel_size = 7 
    bandwidth = 3.0
    blur_type = "Gaussian"
    kernels = fspecial_gaussian(kernel_size, bandwidth)
        
        
    task_current = 'sr'                  # 'sr' for super-resolution
    n_channels = 1 if 'gray' in model_name else 3  # fixed
    pretrained_pth = "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/pretrained_models/model_zoo"
    results = f'Results_DPIR_new/{testset_name}/{model_name}' 
    result_name = testset_name + '_' + task_current + '_' + model_name
    model_path = os.path.join(pretrained_pth, model_name+'.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    script_dir = os.path.dirname(__file__)
    for ssf in test_sf:
        results_dir = os.path.join(script_dir, results, task_current, f"kernel_size_{kernel_size}", f"factor_{ssf}", f"Noise_level_{noise_level}")
        results_dir_true = os.path.join(results_dir, "true")
        results_dir_degraded = os.path.join(results_dir, "degraded")
        results_dir_map = os.path.join(results_dir, "map")
        util.mkdir(results_dir_true)
        util.mkdir(results_dir_degraded)
        util.mkdir(results_dir_map)
    
    dataset_root = "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/Datasets"         
    L_path = os.path.join(dataset_root, testset_name) # L_path, for Low-quality images
    

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(results_dir, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    if 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
    elif 'ircnn' in model_name:
        from models.network_dncnn import IRCNN as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        model25 = torch.load(model_path)
        former_idx = 0

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

   
    test_results_ave = OrderedDict()
    test_results_ave['psnr'] = []
    test_results_ave['lpips'] = []
    test_results_ave['ssim'] = []

    for sf in test_sf:
        border = sf
        modelSigma2 = max(sf, noise_level_model*255.)
        #k_num = 8 if classical_degradation else 1

        # for k_index in range(k_num):
        # logger.info('--------- sf:{:>1d} --k:{:>2d} ---------'.format(sf, k_index))
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['psnr_y'] = []
        test_results['lpips'] = []
        test_results['ssim'] = []

        # if not classical_degradation:  # for bicubic degradation
        #     k_index = sf-2
        k = kernels.cpu().squeeze().numpy()  #[0, k_index].astype(np.float64)

        # util.surf(k) if show_img else None

        Results_dic = OrderedDict()
        
        for idx, img in enumerate(L_paths):

            # --------------------------------
            # (1) get img_L
            # --------------------------------

            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)
            img_H = util.modcrop(img_H, sf)  # modcrop

            if classical_degradation:
                img_L = sr.classical_degradation(img_H, k, sf)
                util.imshow(img_L) if show_img else None
                img_L = util.uint2single(img_L)
            else:
                img_L = util.imresize_np(util.uint2single(img_H), 1/sf)

            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img, img_L.shape) # add AWGN

            # --------------------------------
            # (2) get rhos and sigmas
            # --------------------------------

            rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_model), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1)
            rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

            # --------------------------------
            # (3) initialize x, and pre-calculation
            # --------------------------------

            x = cv2.resize(img_L, (img_L.shape[1]*sf, img_L.shape[0]*sf), interpolation=cv2.INTER_CUBIC)
            if np.ndim(x)==2:
                x = x[..., None]

            if classical_degradation:
                x = sr.shift_pixel(x, sf)
            x = util.single2tensor4(x).to(device)

            img_L_tensor, k_tensor = util.single2tensor4(img_L), util.single2tensor4(np.expand_dims(k, 2))
            [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
            FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)

            # --------------------------------
            # (4) main iterations
            # --------------------------------

            for i in range(iter_num):

                # --------------------------------
                # step 1, FFT
                # --------------------------------

                tau = rhos[i].float().repeat(1, 1, 1, 1)
                x = sr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, sf)

                if 'ircnn' in model_name:
                    current_idx = int(np.ceil(sigmas[i].cpu().numpy()*255./2.)-1)
        
                    if current_idx != former_idx:
                        model.load_state_dict(model25[str(current_idx)], strict=True)
                        model.eval()
                        for _, v in model.named_parameters():
                            v.requires_grad = False
                        model = model.to(device)
                    former_idx = current_idx

                # --------------------------------
                # step 2, denoiser
                # --------------------------------

                if x8:
                    x = util.augment_img_tensor4(x, i % 8)
                    
                if 'drunet' in model_name:
                    x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                    x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)
                elif 'ircnn' in model_name:
                    x = model(x)

                if x8:
                    if i % 8 == 3 or i % 8 == 5:
                        x = util.augment_img_tensor4(x, 8 - i % 8)
                    else:
                        x = util.augment_img_tensor4(x, i % 8)
            
            # --- Metrics ---
            metrics = Metrics(device = device)
            loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
            
            x_map_tensor = x.detach()
            x_map_np = util.tensor2uint(x)
            x_true_tensor = util.single2tensor4(img_H).to(device) / 255.0 #np.transpose(img_H, (2, 0, 1))
            y_tensor = util.single2tensor4(img_L).to(device)
            
            temp_lpips = loss_fn_vgg(x_map_tensor, x_true_tensor).cpu().detach().squeeze().numpy()
            temp_psnr = metrics.psnr_function(x_map_tensor, x_true_tensor)
            temp_ssim = metrics.ssim_function(x_map_tensor, x_true_tensor)
            
            test_results['lpips'].append(temp_lpips)
            test_results['psnr'].append(temp_psnr.cpu().squeeze().numpy())
            test_results['ssim'].append(temp_ssim.cpu().squeeze().numpy())
            
            dic = {"lpips": temp_lpips, "psnr": temp_psnr, "ssim": temp_ssim, "map": x_map_tensor.cpu().squeeze().numpy()}
            Results_dic[img_name] = dic
        
            
            logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB LPIPS: {:.4f} SSIM: {:.3f}'.format(idx+1, img_name+ext, temp_psnr, temp_lpips, temp_ssim))
        
            # --------------------------------
            # (3) img_E
            # --------------------------------
            img_E = util.tensor2uint(x)
            if n_channels == 1:
                img_H = img_H.squeeze()
        
            def save_images(dir, image, name):
                image_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(dir,name+'.png'), image_array)
            save_images(results_dir_degraded, util.single2uint(img_L), "y_" +img_name)
            save_images(results_dir_true, img_H, "x_" +img_name)
            save_images(results_dir_map, img_E, "map_" +img_name)
          
            # --------------------------------
            # (4) img_LEH
            # --------------------------------
            img_L = util.single2uint(img_L).squeeze()
            if n_channels == 3:
                img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                psnr_y  = util.calculate_psnr(img_E_y, img_H_y, border=border)
                test_results['psnr_y'].append(psnr_y)

    # --------------------------------
    # Average PSNR
    # --------------------------------
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('------> Average results of ({}), kernel: ({}) sigma: ({:.2f})'.format(testset_name, blur_type, noise_level_model))
    logger.info('------> Average PSNR : {:.2f} dB'.format(ave_psnr))
    logger.info('------> Average lpips : {:.2f} dB'.format(ave_lpips))
    logger.info('------> Average ssim : {:.2f} dB'.format(ave_ssim))
    test_results_ave['psnr'].append(ave_psnr)
    test_results_ave['lpips'].append(ave_psnr)
    test_results_ave['ssim'].append(ave_psnr)
    
    test_results["ave_psnr"] = ave_psnr
    test_results["ave_lpips"] = ave_lpips
    test_results["ave_ssim"] = ave_ssim
    sio.savemat(os.path.join(results_dir, "Results.mat"), test_results)

if __name__ == '__main__':
    for _ in ["imageffhq_256"]:
        for noise_val in [0.003, 0.01]:
            main(noise_level = noise_val)
