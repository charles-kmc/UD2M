# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""

import distance
import embedding
import io_util
import numpy as np
 

import torch

from collections import defaultdict
import matplotlib.image as mpimg
import pandas as pd
import os

from utils.metrics import *


def compute_cmmd(
    ref_dir, 
    eval_dir, 
    batch_size = 32, 
    max_count = -1
):
  """Calculates the CMMD distance between reference and eval image sets.

  Args:
    ref_dir: Path to the directory containing reference images.
    eval_dir: Path to the directory containing images to be evaluated.
    batch_size: Batch size used in the CLIP embedding calculation.
    max_count: Maximum number of images to use from each directory. A
      non-positive value reads all images available except for the images
      dropped due to batching.

  Returns:
    The CMMD value between the image sets.
  """
  embedding_model = embedding.ClipEmbeddingModel()
  ref_embs = io_util.compute_embeddings_for_dir(
      ref_dir, embedding_model, batch_size, max_count
  )
  eval_embs = io_util.compute_embeddings_for_dir(
      eval_dir, embedding_model, batch_size, max_count
  )
  val = distance.mmd(ref_embs, eval_embs)
  return np.asarray(val)


def cmmd_eval(dir_targets, dir_estimated, batch=16, max_count = 100):
  
  print(
      'The CMMD value is: '
      f' {compute_cmmd(dir_targets, dir_estimated, batch, max_count):.3f}'
  )
   

def psnr_eval(dir_target, dir_estimated):
    # Settings
    eval_metric = Metrics()
    ssim = []
    psnr = []
    
    # Get name of images
    images_name = os.listdir(dir_target)
    
    for name in images_name:
        # Path
        true_pth_name = os.path.join(dir_target, name)
        est_pth_name = os.path.join(dir_estimated, name)
        
        # Load images 
        x_true = torch.from_numpy(mpimg.imread(true_pth_name) / 255.).permute(2,0,1).unsqueeze(0)
        x_est = torch.from_numpy(mpimg.imread(est_pth_name) / 255.).permute(2,0,1).unsqueeze(0)
        
        # Evaluation
        psnr_val = eval_metric.psnr_function(x_true, x_est)
        ssim_val = eval_metric.ssim_function(x_true, x_est)
        
        ssim.append(ssim_val.squeeze().numpy())
        psnr.append(psnr_val.squeeze().numpy())
        
    return (np.round(np.mean(psnr), 2), np.round(np.mean(ssim),2))
    

if __name__ == "__main__":
    # -- root directory
    root_dir = "/home/cmk2000/Documents/Years 2/Python codes and results/Python codes/Codes/Diffusion codes with PnP"
    indices = ["Ours", "DiffPIR", "SGS", "DPS", "DPIR ircnn", "DPIR drunet"]
    # inp
    for dataset_name in ["imagenet_256", "imageffhq_256"]:
        
        dic_inp = defaultdict(lambda:None) 
        dic_inp_psnr = defaultdict(lambda:None) 
        dic_inp["Methods"] = indices
        dic_inp_psnr["Methods"] = indices
        
        for rate in [0.3, 0.5]:
            for noise_val in [0.003]:
                col_name = f"{dataset_name}_rate_{rate}_noise_{noise_val}"
                col_val = []
                col_val_psnr = []
                # -- Ours 
                dir_ours = os.path.join(root_dir, "PnP_ULA Diffusion", f"Results_{dataset_name}", "Inpainting_Results", "Diff_prior_ULA_MASK_random_1.5_Model_diffusion_ffhq_10m_not_equivariant", "27June2024", "Nfe_3", f"Mask_rate_{rate}", f"noise_var_{noise_val}", "alpha_1.0","Rho_op","c_rho_0.1","Maxiter400")
                dir_ours_true = os.path.join(dir_ours, "true")
                dir_ours_mmse = os.path.join(dir_ours, "mmse")
                cmmd_ours = cmmd_eval(dir_ours_true, dir_ours_mmse)
                ssim_psnr_ours = psnr_eval(dir_ours_true, dir_ours_mmse)
                col_val.append(cmmd_ours)                
                col_val_psnr.append(ssim_psnr_ours)                
                
                # -- DiffPIR
                dir_diffpir = os.path.join(root_dir, "DiffPIR_original", f"Results_DiffPIR_inpainting_{dataset_name}", f"Model_name_diffusion_ffhq_10mrate_{1-rate}", "28June2024", f"Noise_level_{noise_val}", "Max_teration_500")
                dir_diffpir_true = os.path.join(dir_diffpir, "true")
                dir_diffpir_mmse = os.path.join(dir_diffpir, "mmse")
                cmmd_diffpir = cmmd_eval(dir_diffpir_true, dir_diffpir_mmse)
                ssim_psnr_diffpir = psnr_eval(dir_diffpir_true, dir_diffpir_mmse)
                col_val.append(cmmd_diffpir)
                col_val_psnr.append(ssim_psnr_diffpir)       
                
                # -- SGS
                dir_sgs = os.path.join(root_dir, "PnP_ULA Diffusion", f"Results_SGS_{dataset_name}", "Inpainting_Results", "MASK_random_1.5_Model_diffusion_ffhq_10m", "10July2024", f"Mask_rate_{1 - rate}", f"noise_var_{noise_val}","Rho_op0.1","Maxiter500")
                dir_sgs_true = os.path.join(dir_sgs, "true")
                dir_sgs_mmse = os.path.join(dir_sgs, "mmse")
                cmmd_sgs = cmmd_eval(dir_sgs_true, dir_sgs_mmse)
                ssim_psnr_sgs = psnr_eval(dir_sgs_true, dir_sgs_mmse)
                col_val.append(cmmd_sgs)
                col_val_psnr.append(ssim_psnr_sgs)       
                
                # -- dps
                dir_dps = os.path.join(root_dir, "diffusion-posterior-sampling", "Results", f"{dataset_name}", "inpainting", "03July2024", f"factor_{1 - rate}", f"Noise_level_{noise_val}", "Maxtimestep_1000")
                dir_dps_true = os.path.join(dir_dps, "true")
                dir_dps_mmse = os.path.join(dir_dps, "mmse")
                cmmd_dps = cmmd_eval(dir_dps_true, dir_dps_mmse)
                ssim_psnr_dps = psnr_eval(dir_dps_true, dir_dps_mmse)
                col_val.append(cmmd_dps)
                col_val_psnr.append(ssim_psnr_dps)       
                
                # -- dpir ircnn
                cmmd_ircnn = 0
                ssim_psnr_ircnn = (0,0)
                col_val.append(cmmd_ircnn)
                col_val_psnr.append(ssim_psnr_ircnn)   
                
                # -- dpir drunet
                cmmd_drunet = 0
                ssim_psnr_drunet = (0,0)
                col_val.append(cmmd_drunet)
                col_val_psnr.append(ssim_psnr_drunet)   
                
                dic_inp[col_name] = col_val
                dic_inp_psnr[col_name] = col_val_psnr
        
        # save dic here as dataframe
        df_inp = pd.DataFrame.from_dict(dic_inp)
        df_inp_psnr = pd.DataFrame.from_dict(dic_inp_psnr)
        # Save to a CSV file in the working directory
        df_inp.to_csv(f'cmmd_results_{dataset_name}_inp.csv')
        df_inp_psnr.to_csv(f'cmmd_results_{dataset_name}_inp_psnr.csv')
                
                
    
    # deb
    for dataset_name in ["imagenet_256", "imageffhq_256"]:
        
        dic_deb = defaultdict(lambda:None)
        dic_deb_psnr = defaultdict(lambda:None)
        dic_deb["Methods"] = indices
        dic_deb_psnr["Methods"] = indices
        k_size = [7, 25]
        for ii, blur in enumerate(["Gaussian", "motion"]):
            if blur == "Gaussian":
                blur0 = "gaussian"
                
            else:
                blur0 = blur
            s = k_size[ii]
            
            for noise_val in [0.003, 0.01]:
                col_name = f"{dataset_name}_{blur}_noise_{noise_val}"
                col_val = []
                col_val_psnr = []
                
                # -- Ours 
                dir_ours = os.path.join(root_dir, "PnP_ULA Diffusion", f"Results_{dataset_name}", "Non_blind_Results", "eta_1", f"SGS_rho_Dif_ULA_{blur0}_1.5_Model_diffusion_ffhq_10m_equivariant", "26June2024", "nfe_3", f"kenerl_size_{s}",  f"noise_var_{noise_val}", "alpha_1.0","Rho_op", "c_rho_5.0", "Maxiter500")
                dir_ours_true = os.path.join(dir_ours, "true")
                dir_ours_mmse = os.path.join(dir_ours, "mmse")
                ssim_psnr_ours = cmmd_eval(dir_ours_true, dir_ours_mmse)
                col_val.append(cmmd_ours)
                col_val_psnr.append(ssim_psnr_ours)
                
                # -- DiffPIR
                dir_diffpir = os.path.join(root_dir, "DiffPIR_original", f"Results_DiffPIR_deb_{dataset_name}", f"blur_{blur}", f"Model_name_diffusion_ffhq_10m", "28June2024", f"Noise_level_{noise_val}", "Max_teration_500")
                dir_diffpir_true = os.path.join(dir_diffpir, "true")
                dir_diffpir_mmse = os.path.join(dir_diffpir, "mmse")
                ssim_psnr_diffpir = cmmd_eval(dir_diffpir_true, dir_diffpir_mmse)
                col_val.append(cmmd_diffpir)
                col_val_psnr.append(ssim_psnr_diffpir)
                
                # -- SGS
                dir_sgs = os.path.join(root_dir, "PnP_ULA Diffusion", f"Results_SGS_{dataset_name}", "Deblurring_Results", f"MASK_{blur}_1.5_Model_diffusion_ffhq_10m", "10July2024", f"noise_var_{noise_val}", "Maxiter100")
                dir_sgs_true = os.path.join(dir_sgs, "true")
                dir_sgs_mmse = os.path.join(dir_sgs, "mmse")
                cmmd_sgs = cmmd_eval(dir_sgs_true, dir_sgs_mmse)
                ssim_psnr_sgs = psnr_eval(dir_sgs_true, dir_sgs_mmse)
                col_val.append(cmmd_sgs)
                col_val_psnr.append(ssim_psnr_sgs)
                
                # -- dps
                dir_dps = os.path.join(root_dir, "diffusion-posterior-sampling", "Results", f"{dataset_name}", f"{blur0}_blur", "03July2024", f"{blur0}_blur_size_{s}", f"Noise_level_{noise_val}", "Maxtimestep_1000")
                dir_dps_true = os.path.join(dir_dps, "true")
                dir_dps_mmse = os.path.join(dir_dps, "mmse")
                cmmd_dps = cmmd_eval(dir_dps_true, dir_dps_mmse)
                ssim_psnr_dps = psnr_eval(dir_dps_true, dir_dps_mmse)
                col_val.append(cmmd_dps)
                col_val_psnr.append(ssim_psnr_dps)
                
                # -- dpir ircnn
                dir_ircnn = os.path.join(root_dir, "DPIR", "Results_DPIR", f"{dataset_name}", "ircnn_color", "deblur", f"{blur0}", f"kernel_size_{s}", f"Noise_level_{noise_val}")
                dir_ircnn_true = os.path.join(dir_ircnn, "true")
                dir_ircnn_map = os.path.join(dir_ircnn, "map")
                cmmd_ircnn = cmmd_eval(dir_ircnn_true, dir_ircnn_map)
                ssim_psnr_ircnn = psnr_eval(dir_ircnn_true, dir_ircnn_map)
                col_val.append(cmmd_ircnn)
                col_val_psnr.append(ssim_psnr_ircnn)
                
                # -- dpir drunet
                dir_drunet = os.path.join(root_dir, "DPIR", "Results_DPIR", f"{dataset_name}", "drunet_color", "deblur", f"{blur0}", f"kernel_size_{s}", f"Noise_level_{noise_val}")
                dir_drunet_true = os.path.join(dir_drunet, "true")
                dir_drunet_map = os.path.join(dir_drunet, "map")
                cmmd_drunet = cmmd_eval(dir_drunet_true, dir_drunet_map)
                ssim_psnr_drunet = psnr_eval(dir_drunet_true, dir_drunet_map)
                col_val.append(cmmd_drunet)
                col_val_psnr.append(ssim_psnr_drunet)
                
                dic_deb[col_name] = col_val
                dic_deb_psnr[col_name] = col_val_psnr
        
        # save dic here as dataframe
        df_deb = pd.DataFrame.from_dict(dic_deb)
        df_deb_psnr = pd.DataFrame.from_dict(dic_deb_psnr)
        # Save to a CSV file in the working directory
        df_deb.to_csv(f'cmmd_results_{dataset_name}_debur.csv')
        df_deb_psnr.to_csv(f'cmmd_results_{dataset_name}_debur_psnr.csv')

    # sr
    for dataset_name in ["imagenet_256", "imageffhq_256"]:
        
        dic_sr = defaultdict(lambda:None)
        dic_sr_psnr = defaultdict(lambda:None)
        dic_sr["Methods"] = indices
        dic_sr_psnr["Methods"] = indices
        
        for factor in [4]:
            for noise_val in [0.003, 0.01]:
                col_name = f"{dataset_name}_factor_{factor}_noise_{noise_val}"
                col_val = []
                col_val_psnr = []
                
                # -- Ours 
                dir_ours = os.path.join(root_dir, "PnP_ULA Diffusion", f"Results_{dataset_name}", "Results_sisr", "SGS_rho_Dif_ULA_1.5_Model_diffusion_ffhq_10m_not_equivariant", "04July2024", "kenerl_size_7",  f"noise_var_{noise_val}", "alpha_1.0","Rho_op", "c_rho_10.0", "Maxiter300")
                dir_ours_true = os.path.join(dir_ours, "true")
                dir_ours_mmse = os.path.join(dir_ours, "mmse")
                cmmd_ours = cmmd_eval(dir_ours_true, dir_ours_mmse)
                ssim_psnr_ours = psnr_eval(dir_ours_true, dir_ours_mmse)
                col_val.append(cmmd_ours)
                col_val_psnr.append(ssim_psnr_ours)
                
                # -- DiffPIR
                dir_diffpir = os.path.join(root_dir, "DiffPIR_original", f"Results_DiffPIR_sisr_{dataset_name}", f"Model_name_diffusion_ffhq_10m", "03July2024","Kernel_7_sigma_3.0", f"Factor_{factor}", f"Noise_level_{noise_val}", "Max_teration_500")
                dir_diffpir_true = os.path.join(dir_diffpir, "true")
                dir_diffpir_mmse = os.path.join(dir_diffpir, "mmse")
                cmmd_diffpir = cmmd_eval(dir_diffpir_true, dir_diffpir_mmse)
                ssim_psnr_diffpir = psnr_eval(dir_diffpir_true, dir_diffpir_mmse)
                col_val.append(cmmd_diffpir)
                col_val_psnr.append(ssim_psnr_diffpir)
                
                # -- SGS
                dir_sgs = os.path.join(root_dir, "PnP_ULA Diffusion", f"Results_SGS_{dataset_name}", "SISR_Results", f"Results_factor_{factor}_1.5_Model_diffusion_ffhq_10m", "10July2024", f"noise_var_{noise_val}", "Rho_op0.1", "Maxiter100")
                dir_sgs_true = os.path.join(dir_sgs, "true")
                dir_sgs_mmse = os.path.join(dir_sgs, "mmse")
                cmmd_sgs = cmmd_eval(dir_sgs_true, dir_sgs_mmse)
                ssim_psnr_sgs = psnr_eval(dir_sgs_true, dir_sgs_mmse)
                col_val.append(cmmd_sgs)
                col_val_psnr.append(ssim_psnr_sgs)
                
                # -- dps
                dir_dps = os.path.join(root_dir, "diffusion-posterior-sampling", "Results", f"{dataset_name}", "super_resolution", "03July2024", f"factor_{factor}", f"Noise_level_{noise_val}", "Maxtimestep_1000")
                dir_dps_true = os.path.join(dir_dps, "true")
                dir_dps_mmse = os.path.join(dir_dps, "mmse")
                cmmd_dps = cmmd_eval(dir_dps_true, dir_dps_mmse)
                ssim_psnr_dps = psnr_eval(dir_dps_true, dir_dps_mmse)
                col_val.append(cmmd_dps)
                col_val_psnr.append(ssim_psnr_dps)
                                
                # -- dpir ircnn
                dir_ircnn = os.path.join(root_dir, "DPIR", "Results_DPIR", f"{dataset_name}", "ircnn_color", "sr", "kernel_size_7", f"factor_{factor}", f"Noise_level_{noise_val}")
                dir_ircnn_true = os.path.join(dir_ircnn, "true")
                dir_ircnn_map = os.path.join(dir_ircnn, "map")
                cmmd_ircnn = cmmd_eval(dir_ircnn_true, dir_ircnn_map)
                ssim_psnr_ircnn = psnr_eval(dir_ircnn_true, dir_ircnn_map)
                col_val.append(cmmd_ircnn)
                col_val_psnr.append(ssim_psnr_ircnn)
                
                # -- dpir drunet
                dir_drunet = os.path.join(root_dir, "DPIR", "Results_DPIR", f"{dataset_name}", "drunet_color", "sr", "kernel_size_7", f"factor_{factor}", f"Noise_level_{noise_val}")
                dir_drunet_true = os.path.join(dir_drunet, "true")
                dir_drunet_map = os.path.join(dir_drunet, "map")
                cmmd_drunet = cmmd_eval(dir_drunet_true, dir_drunet_map)
                ssim_psnr_drunet = psnr_eval(dir_drunet_true, dir_drunet_map)
                col_val.append(cmmd_drunet)
                col_val_psnr.append(ssim_psnr_drunet)
                
                dic_sr[col_name] = col_val
                dic_sr_psnr[col_name] = col_val_psnr
        
        # save dic here as dataframe
        df_sr = pd.DataFrame.from_dict(dic_sr)
        df_sr_psnr = pd.DataFrame.from_dict(dic_sr_psnr)
        # Save to a CSV file in the working directory
        df_sr.to_csv(f'cmmd_results_{dataset_name}_sr.csv')
        df_sr_psnr.to_csv(f'cmmd_results_{dataset_name}_sr_psnr.csv')
    
    
    
