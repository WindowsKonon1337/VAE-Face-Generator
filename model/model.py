import torch
from torch import nn
import torch.nn.functional as F

final_conv_size = 83
kernel_size = 16

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, device = None):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        self.device = 'cpu' if device is None else device

        self.linear = nn.Sequential(
            nn.Linear(32 * final_conv_size ** 2, 128),
            nn.ReLU()
        )

        self.linear_mu = nn.Sequential(
            nn.Linear(128, encoded_space_dim),
            nn.Dropout(0.25)
        )

        self.linear_sigma = nn.Sequential(
            nn.Linear(128, encoded_space_dim),
            nn.Dropout(0.25)
        )
        
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        
        mu = self.linear_mu(x)
        sigma = self.linear_sigma(x)

        N = self.N.sample(mu.shape).to(self.device)

        z = mu + torch.exp(sigma / 2)*N
        return z, mu, sigma
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 32 * final_conv_size**2),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(unflattened_size=(32, final_conv_size, final_conv_size), dim=1)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, kernel_size),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 3, kernel_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device = None):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, device)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        return self.decoder(z), mu, sigma