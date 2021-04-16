import torch
import torch.nn as nn

class Generator(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        nf = image_size
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf*8, out_channels=nf*4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=nf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nf*4, out_channels=nf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=nf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=nf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nf*2, out_channels=nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=nf),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=nf , out_channels=3, kernel_size=4, stride=2, padding=1))

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    
class Discriminator(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        nf = image_size
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nf, out_channels=nf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(in_channels=nf*2, out_channels=nf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=nf*2, out_channels=nf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Sequential(
            nn.Linear((image_size//16)*(image_size//16)*nf*4, 1)
            )
        
    def forward(self, x):
        x = self.main_module(x)
        x = x.view(x.size()[0], -1)
        return torch.sigmoid(self.output(x))