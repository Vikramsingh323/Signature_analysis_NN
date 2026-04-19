import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # A simple Convolutional Neural Network backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), # 1 input channel (grayscale), 32 output channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Fully connected layers to create an embedding
        self.fc = nn.Sequential(
            nn.Linear(128 * 25 * 25, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128) # Final embedding is a 128-dimensional vector
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) # Flatten
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # Pass both images through the same network (twins)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Forces genuine pairs to have a small distance, and forged pairs to have a large distance.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Calculate Euclidean distance between the two image embeddings
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        # Loss formula
        # If label == 1 (Genuine): loss is proportional to distance
        # If label == 0 (Forged): loss is proportional to max(0, margin - distance)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
