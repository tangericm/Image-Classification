import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

def load_CIFAR10(N, input_shape):
    # Define transformations
    C, H, W = input_shape
    transform = transforms.Compose([transforms.Resize((H+20, W+20)),
                                    transforms.RandomCrop((H, W)),
                                    transforms.ToTensor()])

    # Load the training dataset
    subset_size = N 
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    if subset_size > 0:
        full_train, _ = random_split(full_train, [subset_size, len(full_train) - subset_size])

    # Load the test dataset
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split into training (80%) and validation (20%)
    train_size = int(0.8*len(full_train))
    val_size = len(full_train)-train_size
    train, validation = random_split(full_train, [train_size, val_size])
    return train, validation, test