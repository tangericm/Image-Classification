import os
import csv
import logging
from datetime import datetime
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

def set_seed(seed: int = 42) -> None:
    """Sets the seed for reproducibility for PyTorch and NumPy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_CIFAR10(N, input_shape, seed: int = 42):
    # Define transformations
    C, H, W = input_shape
    transform_train = transforms.Compose([transforms.Resize((H, W)),
                                    transforms.RandomCrop(64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.Resize((H, W)),
                                    transforms.ToTensor()])

    # Generator for deterministic behavior
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)

    # Load the training dataset
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    if N < 2:
        raise Exception("N must be at least 2 to create training and validation splits.")
    else:
        full_train, _ = random_split(full_train, [N, len(full_train) - N], generator=g)

    # Load the test dataset
    test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Split into training (80%) and validation (20%)
    train_size = int(0.8*len(full_train))
    val_size = len(full_train)-train_size
    train, validation = random_split(full_train, [train_size, val_size], generator=g)
    return train, validation, test

def setup_logging(run_name: str, log_dir: str = "models"):
    """Create logging directory and configure a logger that writes to console and file.

    Returns: (logger, metrics_csv_path, checkpoint_dir)
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create directory for this run under logs
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Create directory for model checkpoints
    ckpt_dir = os.path.join("models", run_name, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    # File paths
    log_file = os.path.join(ckpt_dir, "training.log")
    metrics_file = os.path.join(ckpt_dir, "metrics.csv")

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