import os
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from time import time


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.beta_gen = 1.0

    def posterior_sample(self, mu_z, logsd_z):
        sd_z = torch.exp(logsd_z)
        eps = torch.randn_like(sd_z)
        return mu_z + eps*sd_z

    def forward(self, x):
        mu_z, logsd_z = self.encoder(x)
        z = self.posterior_sample(mu_z, logsd_z)
        mu_x, gamma_x = self.decoder(z)
        return mu_x, gamma_x, mu_z, logsd_z

    def compute_loss(self, x, mu_x, gamma_x, mu_z, logsd_z):
        BATCH_SIZE = x.size(0)
        HALF_LOG_TWO_PI = 0.91894
        sd_z = torch.exp(logsd_z)

        kl_loss = 0.5*torch.sum(mu_z.pow(2) + sd_z.pow(2) - 1 - 2*logsd_z) / BATCH_SIZE
        gen_loss = torch.sum(0.5*((x - mu_x)/gamma_x).pow(2) + gamma_x.log() + HALF_LOG_TWO_PI) / BATCH_SIZE

        return kl_loss + self.beta_gen * gen_loss

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def fit(self, dataloaders, n_epochs=10, lr=1e-3, dvae_sigma=0, device='cuda', nprint=1, path='.'):

        tic = time()

        self.to(device)
        train_loader, test_loader = dataloaders
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            train_loss = 0.0
            self.train()
            train_iter = iter(train_loader)
            test_iter = iter(test_loader)
            # Training steps
            for i in range(len(train_loader)):
                x = next(train_iter)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = x[0]
                # Load data on device
                x = x.to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if dvae_sigma: # DenoisingVAE
                    x_noisy = x + dvae_sigma/255*torch.randn(*x.shape, device=device)
                    mu_x, gamma_x, mu_z, logsd_z = self.forward(x_noisy)
                else:
                    mu_x, gamma_x, mu_z, logsd_z = self.forward(x)

                loss = self.compute_loss(x, mu_x, gamma_x, mu_z, logsd_z)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()*x.shape[0]

            train_loss /= len(train_loader.dataset)

            # Print statistics every 'nprint' epochs
            if epoch % nprint == nprint-1:

                # Model evaluation (test)
                with torch.no_grad():
                    self.eval()

                    # Test loss
                    test_loss = 0
                    for i in range(len(test_loader)):
                        data = next(test_iter)
                        if isinstance(data, tuple) or isinstance(data, list):
                            data = data[0]
                        data = data.to(device)
                        mu_x, gamma_x, mu_z, logsd_z = self.forward(data)
                        test_loss += data.shape[0]*self.compute_loss(data, mu_x, gamma_x, mu_z, logsd_z).item()

                        # Reconstructions
                        if i == 0:
                            n = min(data.size(0), 8)
                            comparison = torch.cat([data[:n], mu_x[:n]])
                            save_image(comparison.cpu(), os.path.join(path,'reconstruction_' + str(epoch) + '.png'), nrow=n)

                    test_loss /= len(test_loader.dataset)

                    print('[Epoch %d / %d]  Train loss = %.5f  |  Test loss = %.5f  |  gamma_x = %.4f  |  %.2f sec'
                                           % (epoch+1, n_epochs, train_loss, test_loss, gamma_x.item(), time()-tic))

                    # Random samples
                    sample = torch.randn(64, self.latent_dim).to(device)
                    sample = self.decoder(sample)[0].cpu()
                    save_image(sample, os.path.join(path,'sample_' + str(epoch) + '.png'))


        toc = time()
        print('Training finished - Elapsed time: %.2f sec' % (toc-tic))
    
    def evaluate(self, test_loader, device = 'cuda'):
        self.eval()
        test_iter = iter(test_loader)
        test_loss = 0
        MU = []
        for i in range(len(test_loader)):
            print('\r%d/%d' % (i, len(test_loader)), end='')
            data = next(test_iter)
            if isinstance(data, tuple) or isinstance(data, list):
                data = data[0]
            data = data.to(device)
            mu_x, gamma_x, mu_z, logsd_z = self.forward(data)
            MU.append(mu_z)
            test_loss += data.shape[0]*self.compute_loss(data, mu_x, gamma_x, mu_z, logsd_z).item()
        
        print("Computing covariance matrix")
        MU = torch.cat(MU, dim=0).T
        tr = torch.trace(torch.cov(MU))
        test_loss /= len(test_loader.dataset)
        print('Test loss = %.5f' % test_loss, 'Trace = %.5f' % tr)
        return test_loss, tr


class VAEfc(VAE):
    def __init__(self, x_shape, latent_dim, enc_hidden_dims, d_activ = F.relu, return_latent_sd = True):
        super(VAEfc, self).__init__()
        self.x_shape = x_shape
        self.input_dim = np.prod(x_shape)
        self.latent_dim = latent_dim
        self.num_hidden_layers = len(enc_hidden_dims)
        self.d_activ = d_activ

        self.encoder_layers = nn.ModuleList([nn.Linear(self.input_dim, enc_hidden_dims[0])])
        self.encoder_layers.extend([nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i+1]) for i in range(self.num_hidden_layers-1)])
        self.encoder_layers.append(nn.Linear(enc_hidden_dims[-1], self.latent_dim))  # mu_z
        self.encoder_layers.append(nn.Linear(enc_hidden_dims[-1], self.latent_dim))  # logsd_z

        self.decoder_layers = nn.ModuleList([nn.Linear(self.latent_dim, enc_hidden_dims[-1])])
        self.decoder_layers.extend([nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i-1]) for i in range(self.num_hidden_layers-1,0,-1)])
        self.decoder_layers.append(nn.Linear(enc_hidden_dims[0], self.input_dim))  # mu_x
        self.gamma_x = nn.Parameter(torch.ones(1))  # gamma_x

        self.return_latent_sd = return_latent_sd

    def encoder(self, x):
        h = F.relu(self.encoder_layers[0](x.flatten(start_dim=1)))
        for i in range(1,self.num_hidden_layers):
            h = F.relu(self.encoder_layers[i](h))
        mu_z = self.encoder_layers[-2](h)
        logsd_z = self.encoder_layers[-1](h)
        return (mu_z, logsd_z) if self.return_latent_sd else mu_z

    def decoder(self, z):
        h = self.d_activ(self.decoder_layers[0](z))
        for i in range(1,self.num_hidden_layers):
            h = self.d_activ(self.decoder_layers[i](h))
        mu_x = torch.sigmoid(self.decoder_layers[-1](h))
        return mu_x.view(-1,*self.x_shape), self.gamma_x

class DCVAE(VAE):
    def __init__(self, x_shape, latent_dim, d_activ = F.relu, ngf=64):
        super(DCVAE, self).__init__()
        nc = x_shape[0]
        self.latent_dim = x_shape[-3]*x_shape[-2]*x_shape[-1]
        encoder_layers = [
            nn.Conv2d(nc, nc*4,4,2,1,bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(nc*4),
            nn.Conv2d(nc*4, nc*16, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(nc*16),
            nn.Conv2d(nc*16, nc*64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(nc*64),
            nn.Conv2d(nc*64, nc*256, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(nc*256),
            nn.Conv2d(nc*256, latent_dim, 4, 1, 0, bias=False),
            nn.Conv2d(nc*256, latent_dim, 4, 1, 0, bias=False)
        ]

        decoder_layers = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, nc*256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nc*256),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(nc*256, nc*64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc*64),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(nc*64, nc*16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc*16),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(nc*16, nc*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc*4),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(nc*4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64`
        ]
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.gamma_x = nn.Parameter(torch.ones(1))  # gamma_x

    
    def encoder(self, x):
        for l in self.encoder_layers[:-2]:
            x = l(x)
        z = self.encoder_layers[-2](x)
        log_sd = self.encoder_layers[-1](x)
        return z,log_sd
    
    def decoder(self, z):
        for i in range(z.ndim, 4):
            z = z.unsqueeze(-1)
        for l in self.decoder_layers:
            z = l(z)
        return z, self.gamma_x

class DCVAEold(VAE):
    def __init__(self, x_shape, latent_dim, d_activ = F.relu, ngf=64):
        super(DCVAE, self).__init__()
        nc = x_shape[0]
        self.latent_dim = latent_dim
        encoder_layers = [
            nn.Conv2d(nc, ngf,4,2,1,bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf),
            nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf*2),
            nn.Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf*4),
            nn.Conv2d(ngf*4, ngf*8, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf*8),
            nn.Conv2d(ngf*8, latent_dim, 4, 1, 0, bias=False),
            nn.Conv2d(ngf*8, latent_dim, 4, 1, 0, bias=False)
        ]

        decoder_layers = [
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64`
        ]
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.gamma_x = nn.Parameter(torch.ones(1))  # gamma_x

    
    def encoder(self, x):
        for l in self.encoder_layers[:-2]:
            x = l(x)
        z = self.encoder_layers[-2](x)
        log_sd = self.encoder_layers[-1](x)
        return z,log_sd
    
    def decoder(self, z):
        for i in range(z.ndim, 4):
            z = z.unsqueeze(-1)
        for l in self.decoder_layers:
            z = l(z)
        return z, self.gamma_x


class ConvVAE(VAE):
    def __init__(self, x_shape, latent_dim, kernel_size=7, activation='relu', return_latent_sd = True):
        super(ConvVAE, self).__init__()
        self.x_shape = x_shape
        self.im_size = x_shape[1]*x_shape[2]  # [C,H,W] format
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.n_channels = [32,64,128,256] #[32,32,64,64]  # For convolutional layers
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu

        self.return_latent_sd = return_latent_sd

        # Encoder layers
        self.conv_enc1 = nn.Conv2d(x_shape[0], self.n_channels[0], kernel_size, stride=1, padding=kernel_size//2)
        self.batchnorm_enc1 = nn.BatchNorm2d(self.n_channels[0])
        self.conv_enc2 = nn.Conv2d(self.n_channels[0], self.n_channels[1], kernel_size, stride=2, padding=kernel_size//2)
        self.batchnorm_enc2 = nn.BatchNorm2d(self.n_channels[1])
        self.conv_enc3 = nn.Conv2d(self.n_channels[1], self.n_channels[2], kernel_size, stride=2, padding=kernel_size//2)
        self.batchnorm_enc3 = nn.BatchNorm2d(self.n_channels[2])
        self.conv_enc4 = nn.Conv2d(self.n_channels[2], self.n_channels[3], kernel_size, stride=2, padding=kernel_size//2)
        self.batchnorm_enc4 = nn.BatchNorm2d(self.n_channels[3])
        self.fc_enc = nn.Linear(self.im_size*self.n_channels[3]//64, 1024)
        self.fc_mu_z    = nn.Linear(1024, latent_dim)  # mu_z
        self.fc_logsd_z = nn.Linear(1024, latent_dim)  # logsd_z

        # Decoder layers
        self.fc_dec1 = nn.Linear(latent_dim, 1024)
        self.fc_dec2 = nn.Linear(1024, self.im_size*self.n_channels[3]//64)
        self.convtrans_dec1 = nn.ConvTranspose2d(self.n_channels[3], self.n_channels[2], kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        self.batchnorm_dec1 = nn.BatchNorm2d(self.n_channels[2])
        self.convtrans_dec2 = nn.ConvTranspose2d(self.n_channels[2], self.n_channels[1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        self.batchnorm_dec2 = nn.BatchNorm2d(self.n_channels[1])
        self.convtrans_dec3 = nn.ConvTranspose2d(self.n_channels[1], self.n_channels[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        self.batchnorm_dec3 = nn.BatchNorm2d(self.n_channels[0])
        self.convtrans_dec4 = nn.ConvTranspose2d(self.n_channels[0], x_shape[0],  kernel_size, stride=1, padding=kernel_size//2)  # mu_x
        self.batchnorm_dec4 = nn.BatchNorm2d(self.x_shape[0])
        self.gamma_x = nn.Parameter(torch.ones(1))  # gamma_x

    def encoder(self, x):
        h = self.activation(self.batchnorm_enc1(self.conv_enc1(x)))
        h = self.activation(self.batchnorm_enc2(self.conv_enc2(h)))
        h = self.activation(self.batchnorm_enc3(self.conv_enc3(h)))
        h = self.activation(self.batchnorm_enc4(self.conv_enc4(h)))
        h = self.activation(self.fc_enc(h.view(-1,self.im_size*self.n_channels[3]//64)))
        mu_z = self.fc_mu_z(h)
        logsd_z = self.fc_logsd_z(h)
        return (mu_z, logsd_z) if self.return_latent_sd else mu_z

    def decoder(self, z):
        h = self.activation(self.fc_dec1(z))
        h = self.activation(self.fc_dec2(h)).view(-1, self.n_channels[3], self.x_shape[1]//8, self.x_shape[2]//8)  # [C,H,W] format
        h = self.activation(self.batchnorm_dec1(self.convtrans_dec1(h)))
        h = self.activation(self.batchnorm_dec2(self.convtrans_dec2(h)))
        h = self.activation(self.batchnorm_dec3(self.convtrans_dec3(h)))
        mu_x = torch.sigmoid(self.batchnorm_dec4(self.convtrans_dec4(h)))
        return mu_x, self.gamma_x


class TwoStageVAE():
    def __init__(self, vae_stage1, enc_hidden_dims=[512,512,512]):
        #super(TwoStageVAE, self).__init__()

        self.stage1 = vae_stage1
        self.latent_dim = vae_stage1.latent_dim
        self.stage2 = VAEfc([self.latent_dim], self.latent_dim, enc_hidden_dims)

    def fit(self, dataloaders, n_epochs1=10, n_epochs2=10, lr1=1e-3, lr2=1e-2, device='cuda', nprint=1, path='.'):

        print(' -- Training VAE Stage 1 --')
        self.stage1.fit(dataloaders, n_epochs1, lr1, device=device, nprint=nprint, path=path)
        self.stage1.eval()
        self.stage1.freeze()

        print(' -- Training VAE Stage 2 --')

        tic = time()

        self.stage2.to(device)
        train_loader, test_loader = dataloaders
        optimizer = optim.Adam(self.stage2.parameters(), lr=lr2)

        for epoch in range(n_epochs2):  # loop over the dataset multiple times

            train_loss = 0.0
            self.stage2.train()

            # Training steps
            for i, (x,_) in enumerate(train_loader, 0):

                # Load data on device
                x = x.to(device)

                # Train wrt stage1's posterior samples

                with torch.no_grad():
                    mu, logsd = self.stage1.encoder(x)
                    z = self.stage1.posterior_sample(mu, logsd)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                mu_z, gamma_z, mu_u, logsd_u = self.stage2.forward(z)
                loss = self.stage2.compute_loss(z, mu_z, gamma_z, mu_u, logsd_u)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)

            # Print statistics every 'nprint' epochs
            if epoch % nprint == nprint-1:

                # Model evaluation (test)
                with torch.no_grad():
                    self.stage2.eval()

                    print('[Epoch %d / %d]  Train loss = %.5f  |  gamma_z = %.4f  |  %.2f sec'
                                           % (epoch+1, n_epochs2, train_loss, gamma_z.item(), time()-tic))

                    #print(list(self.stage2.parameters()))

                    # Random samples
                    sample_u = torch.randn(64, self.stage2.latent_dim).to(device)
                    sample_z = self.stage2.decoder(sample_u)[0]
                    sample_x = self.stage1.decoder(sample_z)[0].cpu()
                    save_image(sample_x, os.path.join(path,'sample_stage2_' + str(epoch) + '.png'))


        toc = time()
        print('Training finished - Elapsed time: %.2f sec' % (toc-tic))
