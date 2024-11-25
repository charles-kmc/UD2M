import pyiqa    

import torch

from collections import defaultdict
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
import shutil
import os
import lpips
        
from .utils import create_batches

def Fid_evatuation(dir_results, device, mmse_sample = True, last_sample = True, method = "ours"):
    """
    Calculate the Fréchet Inception Distance (FID) between the target and estimated images.

    Parameters:
    dir_results (str): The directory of the ref and mmse.
    device: The device.

    Returns:
    float: The FID value between the target and estimated images.
    """
    
    fid = pyiqa.create_metric("fid").to(device)
    
    dir_ref = os.path.join(dir_results, "ref")  
    batch_dir_ref = create_batches(dir_ref, 4)
    
    out = {}
    if mmse_sample:
        dir_mmse = os.path.join(dir_results, "mmse")
        batch_dir_mmse = create_batches(dir_mmse, 4)
        print("we want to compute the fid of the mmse estimate")
        fid_mmse = fid(batch_dir_ref, batch_dir_mmse) 
        out["fid_mmse"]= fid_mmse
        shutil.rmtree(batch_dir_mmse)
    if last_sample:
        dir_last = os.path.join(dir_results, "last")
        batch_dir_last = create_batches(dir_last, 4)
        print("we want to compute the fid of the last estimate")
        fid_last = fid(batch_dir_ref, batch_dir_last) 
        out["fid_last"]= fid_last
        shutil.rmtree(batch_dir_last)
   
    print("fid computed")
    

    shutil.rmtree(batch_dir_ref)

    
    return out



# if __name__ == "__main__":
#     # -- root directory
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     root_reuslts = ...
#     dic_inp = defaultdict(lambda:None) 
    
#     col_name = f"fid"
#     col_val = []
    
#     fid_ours = Fid_evatuation(root_reuslts, device)
#     print(f"FID Ours: {fid_ours}")
#     col_val.append(fid_ours)  

                    