import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
latent_dim = 100
gen = Generator(latent_dim).to(device)
if os.path.exists("signature_generator.pth"):
    gen.load_state_dict(torch.load("signature_generator.pth", map_location=device, weights_only=True))
    gen.eval()
    
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake = gen(noise).detach().cpu().squeeze()
        fake_arr = ((fake.numpy() + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(fake_arr, mode='L')
        img.save("gan_sample.png")
        print("Sample saved to gan_sample.png")
else:
    print("Generator model not found!")
