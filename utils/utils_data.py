from torch.utils.data import Sampler
from torch import Generator, randperm, int64
    

class RandomSubsetDataSampler(Sampler):
    def __init__(self, data_source, num_points, generator=None):
        """
            A sampler that returns random subsets of the full dataset each epoch, 
            loops through the entire dataset before repeating.
            Args:
                data_source: The dataset to sample from.
                num_points: The number of points to sample each epoch.
                generator: Optional random number generator.
            Returns:
                An iterator that yields indices of the sampled points for training data.s
            
            Example Usage:
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    sampler=RandomSubsetDataSampler(train_set, num_points = 4000),  # To fix the length of each epoch to 4000 images
                    drop_last = True  # To drop the last incomplete batch
                )

        """
        super().__init__()
        self.data_source = data_source
        self.num_points = num_points
        if generator is None:
            generator = Generator()
        self.generator = generator
        self.data_size = len(data_source)
        self.total_epochs_per_dataset = self.data_size // self.num_points
        self.ind = -1
    
    def __len__(self):
        return self.num_points
    
    def __iter__(self):
        self.ind += 1
        if self.ind%self.total_epochs_per_dataset == 0:
            self.indices = randperm(self.data_size, generator=self.generator, dtype=int64).tolist()
            self.ind = 0
        for ind in self.indices[self.ind*self.num_points:self.ind*self.num_points + self.num_points]:
            yield ind



        
    

