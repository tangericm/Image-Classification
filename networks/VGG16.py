import torch
import torch.nn as nn

# Visual Geometry Group (VGG) based on https://arxiv.org/abs/1409.1556
# Modified for smaller input images size (32 x 32)
class VGG16(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Input -> Resized image from 32 x 32 x 3 to 64 x 64 x 3
        C, H, W = input_shape
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(C, 64, kernel_size=3, stride=1, padding=1), # 64 x 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64 x 64 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 x 32 x 64

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 32 x 32 x 128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 32 x 32 x 128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 16 x 16 x 128

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 16 x 16 x 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 16 x 16 x 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 16 x 16 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8 x 8 x 256

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 8 x 8 x 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 8 x 8 x 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 8 x 8 x 512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4 x 4 x 512

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 4 x 4 x 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 4 x 4 x 512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 4 x 4 x 512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2 x 2 x 512

            # Adaptive Average Pooling can be used to match the output size to the original VGG16 network
            # Note: AdaptiveAvgPool2d is non-deterministic on CUDA  
            # nn.AdaptiveAvgPool2d((7,7)) # 7 x 7 x 512 to match original architecture
        )
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 512, 4096), # 1 x 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), # 1 x 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), # 1 x num_classes
        )
    
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        # for i, layer in enumerate(self.features):
        #     x = layer(x)
        #     print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        x = self.features(x)
        x = torch.flatten(x, 1)

        # print(f"Input shape: {x.shape}")
        # for i, layer in enumerate(self.classifier):
        #     x = layer(x)
        #     print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")

        x = self.classifier(x)
        return x