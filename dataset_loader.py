import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SignatureDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.real_dict = {}  # Map: owner_id -> list of real image paths
        self.forge_dict = {} # Map: owner_id -> list of forge image paths
        self.pairs = []
        
        self._load_dataset()
        self._create_pairs()
        
    def _extract_owner_id(self, filename):
        # Filename example: 00102001.png or 00102001_aug_0.png
        # The owner is always the last 3 digits of the core 8-digit string
        base_name = filename.split('_')[0].replace('.png', '').replace('.jpg', '')
        if len(base_name) >= 8:
            return base_name[-3:]
        return "Unknown"

    def _load_dataset(self):
        datasets = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
        
        for ds in datasets:
            real_dir = os.path.join(self.base_dir, ds, 'real')
            forge_dir = os.path.join(self.base_dir, ds, 'forge')
            
            # Load Real
            if os.path.exists(real_dir):
                for f in os.listdir(real_dir):
                    if f.endswith(('.png', '.jpg')):
                        owner_id = self._extract_owner_id(f)
                        if owner_id not in self.real_dict:
                            self.real_dict[owner_id] = []
                        self.real_dict[owner_id].append(os.path.join(real_dir, f))
            
            # Load Forge
            if os.path.exists(forge_dir):
                for f in os.listdir(forge_dir):
                    if f.endswith(('.png', '.jpg')):
                        owner_id = self._extract_owner_id(f)
                        if owner_id not in self.forge_dict:
                            self.forge_dict[owner_id] = []
                        self.forge_dict[owner_id].append(os.path.join(forge_dir, f))

    def _create_pairs(self):
        """Creates an equal number of positive (real-real) and negative (real-forge) pairs."""
        for owner_id, real_images in self.real_dict.items():
            if len(real_images) < 2:
                continue
                
            # Create Positive Pairs (Label 1)
            for i in range(len(real_images)):
                img1 = real_images[i]
                img2 = random.choice(real_images)
                # Try not to pair with itself if possible
                if img1 == img2 and len(real_images) > 1:
                    while img1 == img2:
                        img2 = random.choice(real_images)
                self.pairs.append((img1, img2, 1.0))
            
            # Create Negative Pairs (Label 0)
            if owner_id in self.forge_dict and len(self.forge_dict[owner_id]) > 0:
                forge_images = self.forge_dict[owner_id]
                # Match the number of positive pairs we just made
                for i in range(len(real_images)):
                    img1 = random.choice(real_images)
                    img2 = random.choice(forge_images)
                    self.pairs.append((img1, img2, 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
        
        # Convert to float numpy array and normalize to 0-1
        img1 = np.array(img1, dtype=np.float32) / 255.0
        img2 = np.array(img2, dtype=np.float32) / 255.0
        
        # Add channel dimension (1, H, W) for PyTorch
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, label
