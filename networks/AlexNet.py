import torch
import torch.nn as nn

# AlexNet based on https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
# Modified for smaller input images size (32 x 32)
class AlexNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Input -> Resized image from 3 x 32 x 32 to 3 x 64 x 64
        C, H, W = input_shape
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(C, 64, kernel_size=11, stride=4, padding=2), # 15 x 15 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 7 x 7 x 64

            # Layer 2
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2, groups=2),  # 7 x 7 x 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 3 x 3 x 192

            # Layer 3 
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, groups=2), # 3 x 3 x 384
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2), # 3 x 3 x 256
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=2), # 3 x 3 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # 1 x 1 x 256

            # Adaptive Average Pooling can be used to match the output size to the original AlexNet network
            # Note: AdaptiveAvgPool2d is non-deterministic on CUDA  
            # nn.AdaptiveAvgPool2d((6,6)) # 6 x 6 x 256
        )
        # Fully connected layers
        self.classifier = nn.Sequential(
            # Flatten -> 1 x 1 x 256 = 256
            nn.Dropout(p=0.5),
            nn.Linear(256 * 1 * 1, 4096), 
            nn.ReLU(inplace=True), # 4096

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), 
            nn.ReLU(inplace=True), # 4096

            nn.Linear(4096, num_classes), # 10
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