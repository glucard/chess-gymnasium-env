import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional layers and a skip connection.
    
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions if in_channels != out_channels
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  # Project identity

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Add skip connection
        return F.relu(out)


class CustomResNetxtractor(BaseFeaturesExtractor):
    """
    CNN features extractor using a ResNet-style architecture.
    
    :param observation_space: The observation space.
    :param features_dim: Number of features to extract.
    """
    def __init__(self, observation_space, features_dim=256):
        super(CustomResNetxtractor, self).__init__(observation_space, features_dim)
        
        # We assume the input observation space is (H, W, C) = (8, 8, 12)
        # PyTorch expects (N, C, H, W), so we use 12 input channels
        in_channels = 12

        # Initial "stem" layer
        # Input: (12, 8, 8) -> Output: (64, 8, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        # Stack of residual blocks
        # We keep the 8x8 spatial dimensions
        self.res_blocks = nn.Sequential(
            ResidualBlock(12, 24),      # (64, 8, 8) -> (64, 8, 8)
            ResidualBlock(24, 24),     # (64, 8, 8) -> (128, 8, 8)
            ResidualBlock(24, 48),    # (128, 8, 8) -> (128, 8, 8)
        )
        
        self.flatten = nn.Flatten()

        # Calculate the output size of the ResNet blocks
        with th.no_grad():
            # Dummy tensor with correct (N, C, H, W) shape
            dummy_input = th.rand(1, in_channels, 8, 8)
            x = self.stem(dummy_input)
            x = self.res_blocks(x)
            n_flatten = self.flatten(x).shape[1] # Get features dimension

        print("n_flatten custom resnet:", n_flatten)
        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Permute from (N, H, W, C) to (N, C, H, W)
        x = observations.permute(0, 3, 1, 2) 
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.flatten(x)
        return self.linear(x)