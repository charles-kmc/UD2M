
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
import torch

import numpy as np
from glob import glob
from PIL import Image
import os
import random

import utils as utils

__DATASET__ = {}

def set_seed(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name.lower(), None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name.lower()] = cls
        return cls
    return wrapper

def get_dataset(name: str, root_dataset:str, im_size:int, subset_data:str):
    if __DATASET__.get(name.lower(), None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name.lower()](root_dataset=root_dataset, im_size=im_size, subset_data=subset_data)

def get_dataloader(dataset: VisionDataset,args, subset_data="train"):
    
    sampler = RandomSampler(dataset, 
                            num_samples=args.data.number_sample_per_epoch, 
                            replacement=False
                        ) if args.data.is_random_sampler and subset_data=="train" else None
    
    # set random seed for reproductivity
    if subset_data!="train":
        set_seed()
    dataloader = DataLoader(dataset, 
                            batch_size = args.data.train_batch_size if subset_data=="train" else args.data.test_batch_size, 
                            shuffle =  False, 
                            num_workers = args.data.num_workers,
                            sampler = sampler
                        )
   
    return dataloader

@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root_dataset: str, im_size:int, subset_data="train"):
        super().__init__(root_dataset)
        
        self.transforms = v2.Compose([
            v2.ToTensor(), 
            v2.Resize((im_size, im_size)),
        ])

        self.fpaths = sorted(glob(os.path.join(root_dataset, "FFHQ", f'{subset_data}', f'{subset_data}/*.png'), recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root_dataset."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
    
        if self.transforms is not None:
            img = self.transforms(img)
        img = utils.image_transform(img)
        return img

@register_dataset(name='lsun')
class LSUNDataset(VisionDataset):
    def __init__(self, root_dataset: str, im_size:int, subset_data:str = "train"):
        super().__init__(root_dataset)
        
        self.transforms = v2.Compose([
            v2.ToTensor(), 
            v2.Resize((im_size, im_size)),
        ])

        try:
            lists_paths_dir = f"/users/cmk2000/sharedscratch/Datasets/LSUN/bedroom_{subset_data}_paths.csv"
            if os.path.exists(lists_paths_dir):
                with open(lists_paths_dir, 'r') as f:  
                    self.fpaths = [line.split("\n")[0] for line in f.readlines()]
                self.fpaths = self.fpaths[1:]
            else:
                self.fpaths = sorted(glob(os.path.join(root_dataset,"LSUN", f"bedroom_{subset_data}" + '/*.webp'), recursive=True)) 
        except:
            self.fpaths = sorted(glob(os.path.join(root_dataset + '/*.png'), recursive=True))
            
        print("\n\n", os.path.join(root_dataset,"LSUN", f"bedroom_{subset_data}"))
    
        assert len(self.fpaths) > 0, "File list is empty. Check the root_dataset."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = utils.image_transform(img)
        return img

@register_dataset(name='imagenet')
class ImageNetDataset(VisionDataset):
    def __init__(self, root_dataset: str, im_size:int, subset_data:str = "train"):
        super().__init__(root_dataset)
        
        self.transforms = v2.Compose([
            v2.ToTensor(), 
            v2.Resize((im_size, im_size)),
        ])
        self.fpaths = glob(os.path.join(root_dataset,"ImageNet", f"{subset_data}", f"{subset_data}" + '/*.JPEG')) + glob(os.path.join(root_dataset,"ImageNet", f"{subset_data}", f"{subset_data}" + '/*.png'))
        assert len(self.fpaths) > 0, f"File list is empty. Check the root_dataset. {os.path.join(root_dataset,'ImageNet', f'{subset_data}', f'{subset_data}' + '/*.JPEG')}"

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = utils.image_transform(img)
        return img

@register_dataset(name='MNIST')
class MNISTDataset(VisionDataset):
    def __init__(self, root_dataset: str="datasets/", im_size:int=32, subset_data:str = "train"):
        super().__init__(root_dataset)
        self.transforms = v2.Compose([
            v2.ToTensor(), 
            v2.Pad(2),
        ])

        from torchvision.datasets import MNIST
        sup_set = MNIST(root=root_dataset, train= subset_data in ["train", "val"], download=True)

        if subset_data=="train":
            self.data = sup_set.data[:59000]
        elif subset_data=="val":
            self.data = sup_set.data[59000:]
        else:
            self.data = sup_set.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        img = self.data[index].unsqueeze(0)
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        img = utils.image_transform(img)
        return img

