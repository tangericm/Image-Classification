import torch
import torch.nn as nn

class Inception(nn.Module):
    """
    Inception (GoogLeNet v1) module.

    Structure:
        Branch 1: 1x1 conv
        Branch 2: 1x1 conv -> 3x3 conv
        Branch 3: 1x1 conv -> 5x5 conv
        Branch 4: 3x3 max pool -> 1x1 conv
    """
    def __init__(self, in_channels: int, c1x1: int, c3x3_reduce: int, c3x3: int, c5x5_reduce: int, c5x5: int, pool_proj: int):
        super().__init__()

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3x3_reduce, c3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 1x1 -> 5x5 branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5x5_reduce, c5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # 3x3 max pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3, b4], dim=1)

class GoogLeNet(nn.Module):
    """
    GoogLeNet / Inception v1 backbone adapted for smaller CIFAR-10 images (resized to 64x64).
    Based on https://arxiv.org/abs/1409.4842

    Input:
        input_shape: (C, H, W), e.g. (3, 64, 64)
        num_classes: number of output classes, e.g. 10 for CIFAR-10
    """
    def __init__(self, input_shape, num_classes: int):
        super().__init__()
        C, H, W = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=7, stride=2, padding=3), # B x 64 x 32 x 32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # B x 64 x 16 x 16

            nn.Conv2d(64, 64, kernel_size=1), # B x 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), # B x 192 x 16 x 16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # B x 192 x 8 x 8
        )
        # Inception 3
        self.inception3a = Inception(in_channels=192, c1x1=64, c3x3_reduce=96, c3x3=128, c5x5_reduce=16, c5x5=32, pool_proj=32) # B x 256 x 8 x 8
        self.inception3b = Inception(in_channels=256, c1x1=128, c3x3_reduce=128, c3x3=192, c5x5_reduce=32, c5x5=96, pool_proj=64) # B x 480 x 8 x 8
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # B x 480 x 4 x 4

        # Inception 4
        self.inception4a = Inception(in_channels=480, c1x1=192, c3x3_reduce=96, c3x3=208, c5x5_reduce=16, c5x5=48, pool_proj=64) # B x 512 x 4 x 4
        self.inception4b = Inception(in_channels=512, c1x1=160, c3x3_reduce=112, c3x3=224, c5x5_reduce=24, c5x5=64, pool_proj=64) # B x 512 x 4 x 4
        self.inception4c = Inception(in_channels=512, c1x1=128, c3x3_reduce=128, c3x3=256, c5x5_reduce=24, c5x5=64, pool_proj=64) # B x 512 x 4 x 4
        self.inception4d = Inception(in_channels=512, c1x1=112, c3x3_reduce=144, c3x3=288, c5x5_reduce=32, c5x5=64, pool_proj=64) # B x 528 x 4 x 4
        self.inception4e = Inception(in_channels=528, c1x1=256, c3x3_reduce=160, c3x3=320, c5x5_reduce=32, c5x5=128, pool_proj=128) # B x 832 x 4 x 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # B x 832 x 2 x 2

        # Inception 5
        self.inception5a = Inception(in_channels=832, c1x1=256, c3x3_reduce=160, c3x3=320, c5x5_reduce=32, c5x5=128, pool_proj=128) # B x 832 x 2 x 2
        self.inception5b = Inception(in_channels=832, c1x1=384, c3x3_reduce=192, c3x3=384, c5x5_reduce=48, c5x5=128, pool_proj=128) # B x 1024 x 2 x 2
        # AdaptiveAvgPool is non-deterministic on CUDA
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 1 x 1 x 1024
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(2 * 2 * 1024, num_classes) # 1 x num_classes

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        He (Kaiming) initialization for conv layers, small normal for linear.
        This is robust for deep ReLU networks like GoogLeNet/VGG.
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
        # Initial conv and maxpool layers
        x = self.features(x)

        # Inception 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Classifier
        x = torch.flatten(x, 1)         # (B, 4 * 4 * 1024)
        x = self.dropout(x)
        x = self.fc(x)                  # (B, num_classes)

        return x