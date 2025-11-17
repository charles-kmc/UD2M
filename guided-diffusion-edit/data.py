import torch 
from torchvision.datasets import MNIST
from torchvision import transforms
import os
# Load MNIST data and save each image to a folder of 32x32 png images

mnist_data = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
]))  # Use the last 128 images for val
# Use MPI to load the data in parallel
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Rank {rank} out of {size} is processing data...")
done = 0

# Distribute the data loading across MPI ranks
data_per_rank = 128 // size
start_index = len(mnist_data) - 128 + rank * data_per_rank
end_index =  len(mnist_data) - 128 +(rank + 1) * data_per_rank if rank != size - 1 else len(mnist_data)

os.makedirs('./data/mnist_images/val', exist_ok=True)
for i in range(start_index, end_index):
    if rank == 0:
        print(f"\rProcessing image {i+1}/{data_per_rank}  | {(i+1)/data_per_rank*100}%", end="", flush=True)
    image, label = mnist_data[i]
    image = transforms.ToPILImage()(image)
    image.save(f'./data/mnist_images/val/{label}_{i}.png')
if rank == 0:
    print("MNIST images saved to ./data/mnist_images/val")


###############################################

mnist_data = MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
]))  # Use the last 128 images for val
# Use MPI to load the data in parallel
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Rank {rank} out of {size} is processing data...")
done = 0

# Distribute the data loading across MPI ranks
data_per_rank = (len(mnist_data)-128) // size
start_index = rank * data_per_rank
end_index =  (rank + 1) * data_per_rank if rank != size - 1 else len(mnist_data)-128

os.makedirs('./data/mnist_images/train', exist_ok=True)
for i in range(start_index, end_index):
    if rank == 0:
        print(f"\rProcessing image {i+1}/{data_per_rank}  | {(i+1)/data_per_rank*100}%", end="", flush=True)
    image, label = mnist_data[i]
    image = transforms.ToPILImage()(image)
    image.save(f'./data/mnist_images/train/{label}_{i}.png')
if rank == 0:
    print("MNIST images saved to ./data/mnist_images/train")


###############################################

mnist_data = MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
]))  # Use the last 128 images for val
# Use MPI to load the data in parallel
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(f"Rank {rank} out of {size} is processing data...")
done = 0

# Distribute the data loading across MPI ranks
data_per_rank = (len(mnist_data)) // size
start_index = rank * data_per_rank
end_index =  (rank + 1) * data_per_rank if rank != size - 1 else len(mnist_data)

os.makedirs('./data/mnist_images/test', exist_ok=True)
for i in range(start_index, end_index):
    if rank == 0:
        print(f"\rProcessing image {i+1}/{data_per_rank}  | {(i+1)/data_per_rank*100}%", end="", flush=True)
    image, label = mnist_data[i]
    image = transforms.ToPILImage()(image)
    image.save(f'./data/mnist_images/test/{label}_{i}.png')
if rank == 0:
    print("MNIST images saved to ./data/mnist_images/test")