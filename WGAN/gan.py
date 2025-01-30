import torch 
import torch.nn as nn 

class Genarator(nn.Module):
    def __init__(self, args):
        super(Genarator, self).__init__()
        
        def block(in_features, out_features, normilize=True):
            layers = [nn.Linear(in_features=in_features, out_features=out_features)]
            if normilize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            return layers
        
        input_dim = args.latent_dim + args.n_classes + args.code_dim

        self.init_size = args.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Discriminator model       
class Discriminator(nn.Module):
    def __init__(self, im_size, channels=3):
        super(Discriminator, self).__init__()
        
        # block
        def discriminator_block(in_features, out_features, normilize=True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_features, out_features, 3, 2, 1), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Dropout2d(0.25)
            ]
            if normilize:
                block.append(nn.BatchNorm2d(out_features, 0.8))
            return block
        
        # model
        self.conv_blocks = nn.Sequential( 
            *discriminator_block(channels, 16, normilize=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = im_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            )
        
    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
        
        
        
        
        
        