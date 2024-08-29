import torch
import numpy as np

def W_1d(x, y):
    x = x.sort(dim=0)[0]
    y = y.sort(dim=0)[0]
    return (x - y).square().mean()

class sliced_wasserstein(torch.nn.Module):
    def __init__(self, num_projections, proj_batch_size = None):
        """
            Implementation of sliced Wasserstein loss using num_projections
        """
        super().__init__()
        self.num_projections = num_projections

        # Batch size for projections 
        if proj_batch_size is None:
            self.proj_batch_size = num_projections
        else: 
            self.proj_batch_size = min(proj_batch_size, num_projections)
            if num_projections % proj_batch_size > 0:
                print('WARNING: Projections not divisible by batch size. Truncating number of projections to ', max(num_projections // proj_batch_size,1)*proj_batch_size) 
        
        self.num_batches = self.num_projections // self.proj_batch_size 

    def forward(self, true_data, gen_data):
        ## Flatten input data
        true_data = torch.flatten(true_data, start_dim = 1)
        gen_data = torch.flatten(gen_data, start_dim = 1)

        ## Project data to 1-d space
        true_data_proj = []
        gen_data_proj = []
        for b in range(self.num_batches):
            proj = self._sample_proj_batch(true_data.shape[-1], true_data.device)
            true_data_proj.append(torch.matmul(true_data, proj))
            gen_data_proj.append(torch.matmul(gen_data, proj))

        true_data_proj = torch.cat(true_data_proj, dim = -1)
        gen_data_proj = torch.cat(gen_data_proj, dim = -1)

        ## Compute sliced wasserstein distance
        return  W_1d(true_data_proj, gen_data_proj)

    def _sample_proj_batch(self, dim, device):
        proj = torch.randn((dim, self.proj_batch_size), device = device)
        return proj / (proj.square().sum(dim=0).sqrt())

class l2_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, x_hat):
        return (x - x_hat).square().sum()/x.shape[0]


class conv_sw(torch.nn.Module):
    """
        Convolutional Sliced Wasserstein 
        Adapted from https://github.com/UT-Austin-Data-Science-Group/CSW
        Reduces dimension by half at each layer using convolutions with stride 2
        TODO: Extend beyond powers of 2
    """
    def __init__(self, num_projections, img_shape, base_dim = 8):
        super().__init__()
        C, H, W = img_shape[-3:]  # Channels, Height, Width
        d_min = min(H,W)
        self.num_projections = num_projections
        self.num_conv = int(np.log2(d_min / base_dim)) # Total number of convolutions
        H_base = int(H/2**(self.num_conv))
        W_base = int(W/2**(self.num_conv))
        print(W_base)
        # Add layers
        self.conv_layers = []
        # Initial layer
        self.conv_layers.append(
            torch.nn.Conv2d(
                C,
                self.num_projections,
                kernel_size=2,
                stride=2,
                dilation=1,
                bias=False
            )
        )
        # Deep layers 
        for l in range(self.num_conv - 1):
            self.conv_layers.append(
                torch.nn.Conv2d(
                    self.num_projections,
                    self.num_projections,
                    kernel_size=2,
                    stride=2,
                    dilation=1,
                    groups=self.num_projections,
                    bias=False
                )
            )
        # Final layer 
        self.conv_layers.append(
            torch.nn.Conv2d(
                self.num_projections,
                self.num_projections,
                kernel_size=(H_base, W_base),
                stride=1,
                dilation=1,
                groups=self.num_projections,
                bias=False
            )
        )
        # Remove gradients
        for l in self.conv_layers:
            l.weight.requires_grad = False


    
    def _sample_weights(self):
        # Generate random kernels on the sphere
        for l in self.conv_layers:
            l.weight.data = torch.randn_like(l.weight, requires_grad=False)
            l.weight.data /= l.weight.square().sum(dim=(1,2,3), keepdims = True).sqrt()
    
    def slicer(self, x):
        for l in self.conv_layers:
            x = l(x)
        return x

    def forward(self, x, y):
        # Sample random convolutions
        self._sample_weights()
        x = self.slicer(x)
        y = self.slicer(y)
        return W_1d(x, y)




# Function to compute PSNR between 2 images
# Assumes the first dimension indexes batch number
def PSNR(im1, im2):
    return -10 * (im1-im2).flatten(start_dim=1).square().mean(dim=1).log10()
