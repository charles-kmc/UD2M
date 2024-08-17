from PIL import Image #type: ignore
import os 
import glob
from torch.utils.data import DataLoader, random_split#type: ignore
import torchvision.transforms as transforms #type: ignore
import torch#type: ignore
from degradation_model.utils_deblurs import DeblurringModel
from utils.utils import inverse_image_transform, image_transform

class DatasetsImageNet(object):
    def __init__(
                self, 
                dataset_dir, 
                im_size, 
                transform = None, 
                clip_min = -1.0, 
                clip_max = 1.0
            ):
        
        self.dataset_dir = dataset_dir
        self.im_size = im_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # deblurring model
        self.deblurring_model = DeblurringModel(im_size)

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
            self.ref_images = glob.glob(os.path.join(self.dataset_dir , '*.JPEG'))
        
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
        # get noisy image
        noisy_image, op_image = self.deblurring_model.get_noisy_image(image)
        # rescale image btw -1 and 1
        image_scale = image_transform(image)
        image_scale = torch.clamp(image_scale, self.clip_min, self.clip_max)
        noisy_image = torch.clamp(noisy_image, 0, 1)
        
        return image_scale, noisy_image

# get data loader   
def get_data_loader(
                    dataset_path, 
                    im_size, 
                    batch_size, 
                    prop,
                    num_workers = 0, 
                    shuffle = False
                ):
    assert prop <= 1, f"proportion {prop} should between 0 and 1!!"
    
    dataset = DatasetsImageNet(dataset_path, im_size)
    
    # Define the split sizes
    train_size = int(prop * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
   
    data_loader = DataLoader(
                            train_dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers
                            )
    print(len(dataset), len(data_loader), len(train_dataset))
    
    return data_loader
            
