import pyiqa    

import torch

from collections import defaultdict
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2
import os
import lpips
        
from utils import create_batches

def Fid_evatuation(dir_results, device):
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
    dir_mmse = os.path.join(dir_results, "mmse")
    
    batch_dir_ref = create_batches(dir_mmse, 4)
    batch_dir_mmse = create_batches(dir_ref, 4)
    
    fid_val = fid(batch_dir_ref, batch_dir_mmse) 
    
    return fid_val



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

                    