import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,image_channels = 3, latent_dim = 128):
        super(VAE , self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels , 32 , kernel_size=4 , stride=2 , padding=1),
            nn.ReLU(),
            nn.Conv2d(32 , 64  , kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64 , 128 , kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128 , 256 , kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        
        self.flatten_dim = 256*8*8
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.flatten_dim , latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim , latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim , self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 , 128 , kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128 , 64 , kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64 , 32 , kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32 , 3 , kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self , x):   
        x = self.encoder(x)
        x = x.view(x.size(0) , -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu , logvar
    
    def reparameterize(self , mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu+eps*std
    
    def decode(self , z):
        x = self.fc_decode(z)
        x = x.view(x.size(0) , 256 , 8 , 8)
        x = self.decoder(x)
        return x
    
    def forward(self , x):
        mu,logvar = self.encode(x)
        z = self.reparameterize(mu=mu , logvar=logvar)
        x_recon = self.decode(z)
        
        return x_recon,mu,logvar