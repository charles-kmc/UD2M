import pyiqa    
import shutil
import os
import numpy as np
import torch
from .utils import create_batches
from utils import save_images
def Fid_evatuation_numpy(dir_results, device, mmse_sample = True, last_sample = False, init_sample = False, method = "ours"):
    """
    Calculate the Fréchet Inception Distance (FID) between the target and estimated images.

    Parameters:
    file_results (str): The path to the results file.
    device: The device.

    Returns:
    float: The FID value between the target and estimated images.
    """
    from metrics.vae import VAEfc
    from torchvision.transforms import CenterCrop
    vae = VAEfc(
        x_shape=(28, 28), latent_dim=12, enc_hidden_dims=[512, 512], d_activ=torch.nn.functional.relu, return_latent_sd=False
    ).to(device)
    transforms = torch.nn.Sequential(CenterCrop(28), torch.nn.Flatten())
    vae.load_state_dict(torch.load('/users/js3006/unrolled_cgan/models/saved/MNIST_VAE.pth.tar'))

    def embedding(x):
        x = (x - x.min()) / (x.max() - x.min()+1e-8)
        x_t = transforms(x)
        return vae.encoder(x_t)
    
    from metrics.my_metrics import CFID_metric_test
    fid = CFID_metric_test(embedding, 12, device)
    print(dir_results)
    dir_ref = os.path.join(dir_results, "ref", "ref_samples.npy")  
    ref_ims = np.load(dir_ref)
    obs_ims = np.load(os.path.join(dir_results, "y", "y_samples.npy"))

    
    out = {}
    if mmse_sample:
        dir_mmse = os.path.join(dir_results, "mmse", "mmse_samples.npy")
        mmse_ims = np.load(dir_mmse)

        print("we want to compute the fid of the mmse estimate")
        fid_mmse = fid(ref_ims, mmse_ims, obs_ims)
        out["fid_mmse"]= fid_mmse["FID"]
        out["cfid_mmse"]= fid_mmse["CFID"]
    if last_sample:
        dir_last = os.path.join(dir_results, "last")
        last_ims = np.load(os.path.join(dir_last, "last_samples.npy"))
        fid_last = fid(ref_ims, last_ims, obs_ims)
        out["fid_last"]= fid_last["FID"]
        out["cfid_last"]= fid_last["CFID"]
    else:
        out["fid_last"]= 0

    return out


def Fid_evatuation(dir_results, device, mmse_sample = True, last_sample = False, init_sample = False, method = "ours", numpy = False):
    """
    Calculate the Fréchet Inception Distance (FID) between the target and estimated images.

    Parameters:
    dir_results (str): The directory of the ref and mmse.
    device: The device.

    Returns:
    float: The FID value between the target and estimated images.
    """
    
    fid = pyiqa.create_metric("fid").to(device)
    print(dir_results)
    dir_ref = os.path.join(dir_results, "ref")  
    print(dir_ref)
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
    else:
        out["fid_last"]= 0
    if init_sample:
        dir_last = os.path.join(dir_results, "init")
        batch_dir_init = create_batches(dir_last, 4)
        print("we want to compute the fid of the last estimate")
        fid_init = fid(batch_dir_ref, batch_dir_init) 
        out["fid_init"]= fid_init
        shutil.rmtree(batch_dir_init)
    
        
    print("fid computed")
    shutil.rmtree(batch_dir_ref)

    return out
