from torch.utils.data import Dataset 
from torchvision.transforms import Compose
from PIL import Image
import os 

class FFHQ(Dataset):
    def __init__(self, data_folder='../sharedscratch/data/ffhq256x256/', indices = [i for i in range(70000)], transforms = [], train = None):
        super().__init__()
        if train is not None:
            if train:
                indices = [i for i in range(50000,69000)]
            else: 
                indices = [i for i in range(69000,70000)]
        self.data_root = data_folder 
        self.indices = indices  # Can be used to split into training and validation sets
        self.size = len(indices)
        self.transforms = transforms
        if not os.path.exists(data_folder):
            raise ValueError(f"Path {p} does not exist".format(data_folder))

    def __getitem__(self, index):
        index = self.indices[index]
        dir = os.path.join(self.data_root, str(1000*(index//1000)))
        im = Image.open(os.path.join(dir, str(index).zfill(5) + '.png'), formats = ["png"])
        # Transform image
        for T in self.transforms:
            im = T(im)
        return im
    
    def __len__(self):
        return self.size

class DataDir(Dataset):
    def __init__(self, data_folder = 'data/CBSD300', transforms = [], loops = 1):
        # loops = number loops through dataset (can be combined with random crops of small patches)
        super().__init__()
        self.paths = os.listdir(data_folder)
        for i, p in enumerate(self.paths):
            self.paths[i] = os.path.join(data_folder, p)
        self.size = len(self.paths) * int(loops)
        self.transform = Compose(transforms) 
    
    def __getitem__(self, index):
        im = Image.open(self.paths[index%len(self.paths)], formats = ['png'])
        # Apply transforms 
        return self.transform(im)
    
    def __len__(self):
        return self.size