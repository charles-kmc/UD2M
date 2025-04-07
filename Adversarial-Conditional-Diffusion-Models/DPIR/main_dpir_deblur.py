import os.path
import cv2
import logging

import numpy as np
import scipy.io as sio
from datetime import datetime
from collections import OrderedDict
#import hdf5storage #
from scipy import ndimage

import torch

from DPIR.utils import utils_deblur
from DPIR.utils import utils_logger
from DPIR.utils import utils_model
from DPIR.utils import utils_pnp as pnp
from DPIR.utils import utils_sisr as sr
from DPIR.utils import utils_image as util
from DPIR.utils.utils import fspecial_gaussian

from DPIR.metrics import *
import lpips 

def main(noise_level = 0.01, blur_type = "Gaussian", ffhq = 1):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    testset_name = "data" #"imageffhq_256" if ffhq else "imagenet_256"
    noise_level_img = noise_level        # 7.65/255.0         # default: 0, noise level for LR image
    noise_level_model = noise_level_img  # noise level of model, default 0
    model_name = 'ircnn_color'           # 'drunet_gray' | 'drunet_color' | 'ircnn_gray' | 'ircnn_color'
    x8 = True                            # default: False, x8 to boost performance
    iter_num = 8                         # number of iterations
    modelSigma1 = 49
    modelSigma2 = noise_level_model*255.

    show_img = False                     # default: False
    save_L = True                        # save LR image
    save_E = True                        # save estimated image
    save_LEH = False                     # save zoomed LR, E and H images
    border = 0

    # --------------------------------
    # load kernel
    # --------------------------------
    kernel_size = 7 
    bandwidth = 3.0
    if blur_type == "Gaussian":
        kernels = fspecial_gaussian(kernel_size, bandwidth) #hdf5storage.loadmat(os.path.join('kernels', 'Levin09.mat'))['kernels']
    elif blur_type == "motion":
        pth = "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/Datasets/kernels"
        pth_kernels = os.path.join(pth, "kernels.mat")
        kernels = torch.from_numpy(sio.loadmat(pth_kernels)["4"])
    else:
        assert 1==0, "Blur type not implemented"
    kernel_size = kernels.shape[-1]
    
    sf = 1
    task_current = 'deblur'              # 'deblur' for deblurring
    n_channels = 3 if 'color' in  model_name else 1  # fixed
    pretrained_pth = "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/pretrained_models/model_zoo"
    results = f'Results_DPIR_new/{testset_name}/{model_name}'                  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name
    model_path = os.path.join(pretrained_pth, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, results, task_current, blur_type,  f"kernel_size_{kernel_size}", f"Noise_level_{noise_level}")
    results_dir_true = os.path.join(results_dir, "true")
    results_dir_degraded = os.path.join(results_dir, "degraded")
    results_dir_map = os.path.join(results_dir, "map")
    util.mkdir(results_dir_true)
    util.mkdir(results_dir_degraded)
    util.mkdir(results_dir_map)
    
    dataset_root = "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/Datasets"         
    L_path = os.path.join(dataset_root, testset_name) # L_path, for Low-quality images
    

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(results_dir, logger_name+f'{noise_level}.log'))
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
    test_results_ave['psnr'] = []  # record average PSNR for each kernel
    test_results_ave['lpips'] = []  # record average PSNR for each kernel
    test_results_ave['ssim'] = []  # record average PSNR for each kernel

    # for k_index in range(kernels.shape[1]):

    #logger.info('-------k:{:>2d} ---------'.format(k_index))
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['lpips'] = []
    test_results['ssim'] = []
    k = kernels.cpu().squeeze().numpy() #[0, k_index].astype(np.float64)
    #util.imshow(k) if show_img else None

    Results_dic = OrderedDict()
    psnr_mat = []
    lpips_mat = []
    ssim_mat = []
    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_L
        # --------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        img_H = util.modcrop(img_H, 8)  # modcrop

        img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')
        util.imshow(img_L) if show_img else None
        img_L = util.uint2single(img_L)

        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img, img_L.shape) # add AWGN

        # --------------------------------
        # (2) get rhos and sigmas
        # --------------------------------

        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_model), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        # --------------------------------
        # (3) initialize x, and pre-calculation
        # --------------------------------

        x = util.single2tensor4(img_L).to(device)

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
            x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)

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
        x_true_tensor = util.single2tensor4(img_H).to(device)/255.0 #np.transpose(img_H, (2, 0, 1))
        y_tensor = util.single2tensor4(img_L).to(device)
        temp_lpips = loss_fn_vgg(x_map_tensor, x_true_tensor).cpu().detach().squeeze().numpy()
        temp_psnr = metrics.psnr_function(x_map_tensor, x_true_tensor)
        temp_ssim = metrics.ssim_function(x_map_tensor, x_true_tensor)
        # print(x_map_tensor.max(), x_map_tensor.min())
        # print(x_true_tensor.max(), x_true_tensor.min())
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
    for ffhq in [0]:
        for noise_val in [0.003, 0.01]:
            for blur_type in ["Gaussian", "motion"]:
                main(noise_level = noise_val, ffhq = ffhq, blur_type = blur_type)
