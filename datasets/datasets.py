from PIL import Image 
import os 
import glob
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms 
import torch
from degradation_model.utils_deblurs import DeblurringModel
from utils.utils import inverse_image_transform, image_transform

class Datasets(object):
    def __init__(
                self, 
                dataset_dir, 
                im_size, 
                dataset_name = "ImageNet",
                transform = None, 
                clip_min = -1.0, 
                clip_max = 1.0, 
                transform_y = True,
                
            ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.im_size = im_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # # deblurring model
        # self.deblurring_model = DeblurringModel(im_size, transform_y = transform_y)

        # transformer
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.im_size, self.im_size)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform    
        
        # get all images path
        if self.dataset_dir == "" or self.dataset_dir is None or not os.path.exists(self.dataset_dir):
            raise ValueError(f"The dataset directory {self.dataset_dir} is empty!!")
        else:
            if dataset_name == "ImageNet":
                self.ref_images = glob.glob(os.path.join(self.dataset_dir , '*.JPEG'))
            elif dataset_name == "FFHQ":
                self.ref_images = glob.glob(os.path.join(self.dataset_dir , '*.png'))
            else:
                raise ValueError(f"This dataset {dataset_name} is not yet implemented !!")
        
        self.transform_y = transform_y  
         
    def __len__(self):
        """ get the length of the dataset
        """
        return len(self.ref_images)

    def __getitem__(self, idx):
        """ get an item from the dataset
        """
        img_path = self.ref_images[idx]
        image = Image.open(img_path, mode='r').convert('RGB')
        # apply transform
        if self.transform:
            image = self.transform(image)
        # if self.dataset_name == "ImageNet":
        #     # get noisy image
        #     if self.transform_y:
        #         image_scale = image_transform(image)
        #         noisy_image, op_image = self.deblurring_model.get_noisy_image(image_scale)
        #     else:
        #         image_scale = image_transform(image)
        #         noisy_image, op_image = self.deblurring_model.get_noisy_image(image)
            
        #     return image_scale, noisy_image, op_image
        
        # elif self.dataset_name == "FFHQ":
        
        return image_transform(image)
        

# get data loader   
def get_data_loader(
                    dataset_path, 
                    im_size, 
                    batch_size, 
                    prop,
                    num_workers = 0, 
                    shuffle = True
                ):
    assert prop <= 1, f"proportion {prop} should between 0 and 1!!"
    
    dataset = Datasets(dataset_path, im_size)
    
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
                            num_workers=num_workers
                            )
    
    testloader = DataLoader(
                            test_dataset, 
                            batch_size=8, 
                            shuffle=shuffle, 
                            num_workers=num_workers
                            )
    return data_loader, testloader
            
