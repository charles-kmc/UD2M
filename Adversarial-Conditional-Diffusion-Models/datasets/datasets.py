from PIL import Image 
import numpy as np
import os 
import glob
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch
import tifffile as tiff
from pathlib import Path
import pandas as pd

import utils as utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


def fast_scan_webps(root_dir):
    webp_paths = []
    stack = [root_dir]
    pth = "/users/cmk2000/sharedscratch/Datasets/LSUN/infos"
    os.makedirs(pth, exist_ok=True)
    df = pd.DataFrame(columns=["img_path"])
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.name.lower().endswith(".webp"):
                        #webp_paths.append(Path(entry.path))
                        temp_df = pd.DataFrame({"img_path": [Path(entry.path)]})
                        save_dir = os.path.join(pth,"lsun_bedroom_paths.csv")
                        temp_df.to_csv(save_dir, mode='a', header=not os.path.exists(save_dir))
    
                        # df_new = pd.DataFrame({"img_path": webp_paths})
                        # df = pd.concat([df, df_new], ignore_index=True)
                        # df.to_csv(os.path.join(pth,"lsun_bedroom_infos.csv"), index=False)
        except Exception as e:
            print(f"Error scanning {current}: {e}")

    return webp_paths

class GetDatasets:
    def __init__(
                self, 
                dataset_dir, 
                im_size, 
                dataset_name = "FFHQ",
                transform = None, 
                clip_min = -1.0, 
                clip_max = 1.0,  
                type_ = "train"              
            ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.im_size = im_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.type = type_
        
        # transformer
        self.transform = v2.Compose([
            v2.ToTensor(), 
            v2.Resize((self.im_size, self.im_size)),
            #v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # get all images path
        if self.dataset_dir == "" or self.dataset_dir is None or not os.path.exists(self.dataset_dir):
            raise ValueError(f"The dataset directory {self.dataset_dir} is empty!!")
        else:
            if dataset_name == "ImageNet":
                self.ref_images = glob.glob(os.path.join(self.dataset_dir , '*.JPEG'))
            elif dataset_name == "FFHQ":
                self.ref_images = glob.glob(os.path.join(self.dataset_dir , '*.png'))
            elif dataset_name == "LSUN":
                if self.type == "train":
                    data_dir = os.path.join(self.dataset_dir, "train")
                    data_df = pd.read_csv(os.path.join(data_dir.rsplit("/",1)[0],"infos/lsun_bedroom_paths.csv"))
                    self.ref_images = data_df["img_path"].values
                elif self.type == "val":
                    data_dir = os.path.join(self.dataset_dir, "val")
                    self.ref_images = glob.glob(os.path.join(data_dir, '*.png'))
                else:
                    raise ValueError(f"This type {self.type} is not yet implemented !!")
                print(f"Number of images in {dataset_name} dataset: {len(self.ref_images)}")
                # self.ref_images = fast_scan_webps(Path(data_dir))
            elif dataset_name == "CT":
                self.ref_images = glob.glob(os.path.join(self.dataset_dir , '*.tif'))
            else:
                raise ValueError(f"This dataset {dataset_name} is not yet implemented !!")
        
    def __len__(self): 
        """ get the length of the dataset
        """
        return len(self.ref_images)

    def __getitem__(self, idx):
        """ get an item from the dataset 
        """
        img_path = self.ref_images[idx]
        if self.dataset_name=="CT":
            image = tiff.imread(img_path)
            if len(image.shape) == 2:  
                image = np.stack((image,) * 3, axis=-1)  
            elif len(image.shape) == 3 and image.shape[-1] == 3:
                image = image  
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            image = Image.open(img_path, mode='r').convert('RGB')
        
        # apply transform
        image = self.transform(image)
        if self.dataset_name=="CT":
            pass
        else:
            image = utils.image_transform(image)
        
        return image
        

# get data loader   
def get_data_loader(
                    dataset_path, 
                    im_size, 
                    batch_size, 
                    prop,
                    num_workers = 2, 
                    shuffle = True
                ):
    assert prop <= 1, f"proportion {prop} should between 0 and 1!!"
    
    dataset = GetDatasets(dataset_path, im_size)
    
    # Define the split sizes
    train_size = int(prop * len(dataset))
    val_size = len(dataset) - train_size
    val_size = val_size - 16

    # Split the dataset
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, 16, val_size])
   
    data_loader = DataLoader(
                            train_dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers,
                            pin_memory=True
                            )
    
    testloader = DataLoader(
                            test_dataset, 
                            batch_size=8, 
                            shuffle=shuffle, 
                            num_workers=num_workers,
                            pin_memory=True
                            )
    return data_loader, testloader

           