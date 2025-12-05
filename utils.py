import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import logging
import os
import csv
from datetime import datetime

def load_CIFAR10(N, input_shape):
    # Define transformations
    C, H, W = input_shape
    transform = transforms.Compose([transforms.Resize((H+10, W+10)),
                                    transforms.RandomCrop((H, W)),
                                    transforms.ToTensor()])

    # Load the training dataset
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    if N < 2:
        raise Exception("N must be at least 2 to create training and validation splits.")
    else:
        full_train, _ = random_split(full_train, [N, len(full_train) - N])

    # Load the test dataset
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split into training (80%) and validation (20%)
    train_size = int(0.8*len(full_train))
    val_size = len(full_train)-train_size
    train, validation = random_split(full_train, [train_size, val_size])
    return train, validation, test

def setup_logging(run_name: str, log_dir: str = "logs"):
    """Create logging directory and configure a logger that writes to console and file.

    Returns: (logger, metrics_csv_path, checkpoint_dir)
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{timestamp}"
    log_file = os.path.join(log_dir, f"{run_name}.log")
    metrics_file = os.path.join(log_dir, f"{run_name}_metrics.csv")
    ckpt_dir = os.path.join("models", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    # remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create metrics csv with header
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    return logger, metrics_file, ckpt_dir