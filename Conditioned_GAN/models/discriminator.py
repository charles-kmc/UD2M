import torch 
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        
        if self.in_channels!=self.out_channels:
            self.out_channels = self.in_channels
        
        self.input_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1,1))
        self.layers = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(self.out_chans),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), padding=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_channels, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_channels, height, width]
        """
        output = input
        return self.layers(output) + self.input_conv(output)

class FullDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels in the output.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(self.out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.downsample(input)  

    def __repr__(self):
        return f'AvgPool(in_chans={self.in_channels}, out_chans={self.out_channels}\nResBlock(in_chans={self.out_channels}, out_chans={self.out_channels}'

class DiscriminatorModel(nn.Module):
    def __init__(self, im_size=256, in_channels=3, out_channels=1):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
        """
        super().__init__()

        self.in_channels = in_channels*2 # because we will concatenate x and y
        self.out_channels = out_channels
        self.im_size = im_size

        # --- 
        self.initial_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), padding=1),  # 384x384
            nn.LeakyReLU()
        )

        self.encoder_layers = nn.ModuleList()
        self.encoder_layers += [FullDownBlock(32, 64)]      # im_size/2**1
        self.encoder_layers += [FullDownBlock(64, 128)]     # im_size/2**2
        self.encoder_layers += [FullDownBlock(128, 256)]    # im_size/2**3
        self.encoder_layers += [FullDownBlock(256, 128)]    # im_size/2**4
        self.encoder_layers += [FullDownBlock(128, 64)]    # im_size/2**5

        self.dense_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * int(self.im_size/2**5)**2, self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, input, y):
        assert input.shape[-1]==y.shape[-1]==self.im_size, "Dimension should be the same."
        output = torch.cat([input, y], dim=1)
        output = self.initial_layers(output)

        # Apply down-sampling layers
        for layer in self.encoder_layers:
            output = layer(output)
        return self.dense_layer(output)
        