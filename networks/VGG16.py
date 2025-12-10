import torch
import torch.nn as nn


class VGG16(nn.Module):
    """
    Visual Geometry Group (VGG) 16-layer network adapted for smaller CIFAR-10 images (resized to 64x64).
    Based on https://arxiv.org/abs/1409.1556

    Input:
        input_shape: (C, H, W), e.g. (3, 64, 64)
        num_classes: number of output classes, e.g. 10 for CIFAR-10
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()
        C, H, W = input_shape
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(C, 64, kernel_size=3, stride=1, padding=1), # B x 64 x 64 x 64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # B x 64 x 64 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # B x 64 x 32 x 32

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # B x 128 x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # B x 128 x 32 x 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # B x 128 x 16 x 16

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # B x 256 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # B x 256 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # B x 256 x 16 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # B x 256 x 8 x 8

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # B x 512 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # B x 512 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # B x 512 x 8 x 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # B x 512 x 4 x 4

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # B x 512 x 4 x 4
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # B x 512 x 4 x 4
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # B x 512 x 4 x 4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # B x 512 x 2 x 2

            # Adaptive Average Pooling can be used to match the output size to the original VGG16 network
            # Note: AdaptiveAvgPool2d is non-deterministic on CUDA  
            # nn.AdaptiveAvgPool2d((7,7)) # B x 512 x 7 x 7 to match original architecture
        )
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2 * 2 * 512, 4096), # B x 4096
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), # B x 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes), # B x num_classes
        )

        self._init_weights()

    def _init_weights(self):
        """
        He (Kaiming) initialization for conv layers, small normal for linear.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x