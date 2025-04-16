import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN)
    As proposed in: "Image Super-Resolution Using Deep Convolutional Networks"
    Link: https://arxiv.org/abs/1501.00092
    """
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # Feature extraction layer (f1 = 9, n1 = 64)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        # Non-linear mapping layer (f2 = 5, n2 = 32) 
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        # Reconstruction layer (f3 = 5, n3 = num_channels)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x