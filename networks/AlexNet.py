import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet backbone adapted for smaller CIFAR-10 images (resized to 64x64).
    Based on https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

    Input:
        input_shape: (C, H, W), e.g. (3, 64, 64)
        num_classes: number of output classes, e.g. 10 for CIFAR-10
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()
        C, H, W = input_shape
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(C, 64, kernel_size=11, stride=4, padding=2), # B x 64 x 15 x 15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # B x 64 x 7 x 7

            # Layer 2
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2, groups=2),  # B x 192 x 7 x 7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # B x 192 x 3 x 3

            # Layer 3 
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, groups=2), # B x 384 x 3 x 3
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2), # B x 256 x 3 x 3
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=2), # B x 256 x 3 x 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # B x 256 x 1 x 1

            # Adaptive Average Pooling can be used to match the output size to the original AlexNet network
            # Note: AdaptiveAvgPool2d is non-deterministic on CUDA  
            # nn.AdaptiveAvgPool2d((6,6)) # B x 256 x 6 x 6
        )
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 1 * 1, 4096), # B x 4096
            nn.ReLU(inplace=True), 

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), # B x 4096
            nn.ReLU(inplace=True), 

            nn.Linear(4096, num_classes), # B x 10
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