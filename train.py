import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks import *
from utils import load_CIFAR10, setup_logging, set_seed

@dataclass
class TrainConfig:
    model_name: str                  # "AlexNet", "VGG16", "VGG19"
    input_shape: Tuple[int, int, int]  # (C, H, W), e.g. (3, 64, 64)
    num_classes: int = 10
    N: int = 0                       # 0 = use full CIFAR-10
    num_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainHistory:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]
    test_acc: float
    run_name: str = ""
    metrics_file: str = ""
    ckpt_dir: str = ""


MODEL_REGISTRY = {
    "AlexNet": AlexNet,
    "VGG16": VGG16,
    "VGG19": VGG19,
    "GoogLeNet": GoogLeNet,
}


def build_model(config: TrainConfig) -> nn.Module:
    if config.model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name: {config.model_name}")
    model_cls = MODEL_REGISTRY[config.model_name]
    model = model_cls(input_shape=config.input_shape,
                      num_classes=config.num_classes)
    return model


def train_model(
    config: TrainConfig,
    epoch_callback: Optional[Callable[[int, TrainHistory], None]] = None,
) -> TrainHistory:
    """
    Train a model on CIFAR-10 with the given configuration.

    Args:
        config: Training configuration.
        epoch_callback: Optional function called after each epoch:
            epoch_callback(epoch_idx, history_so_far)

    Returns:
        TrainHistory with per-epoch train/val metrics and final test accuracy.
    """
    set_seed(config.seed)

    # Create a human-readable run name (base) for logging
    logger, metrics_file, ckpt_dir = setup_logging(config.model_name)
    logger.info("Starting training run: %s", logger.name)
    # Log hyperparameters
    hparams = {
        "model_name": config.model_name,
        "input_shape": config.input_shape,
        "num_classes": config.num_classes,
        "N": config.N,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": str(config.device),
    }
    logger.info("Hyperparameters: %s", hparams)


    device = torch.device(config.device)
    model = build_model(config).to(device)

    # Load data
    train_ds, val_ds, test_ds = load_CIFAR10(config.N, config.input_shape, seed=config.seed)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    history = TrainHistory(
        train_loss=[],
        train_acc=[],
        val_loss=[],
        val_acc=[],
        test_acc=0.0,
        run_name=logger.name,         
        metrics_file=metrics_file,
        ckpt_dir=ckpt_dir,
    )

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")

    for epoch in range(config.num_epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100.0 * correct / total

        # ---- Validation ----
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total

        history.train_loss.append(epoch_train_loss)
        history.train_acc.append(epoch_train_acc)
        history.val_loss.append(epoch_val_loss)
        history.val_acc.append(epoch_val_acc)

        # ---- Logging ----
        logger.info(
            "Epoch [%d/%d] - Train Loss: %.4f, Train Acc: %.2f%% - "
            "Val Loss: %.4f, Val Acc: %.2f%%",
            epoch + 1,
            config.num_epochs,
            epoch_train_loss,
            epoch_train_acc,
            epoch_val_loss,
            epoch_val_acc,
        )
        
        # Append to metrics CSV
        with open(history.metrics_file, "a", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{epoch_train_loss:.6f}",
                f"{epoch_train_acc:.4f}",
                f"{epoch_val_loss:.6f}",
                f"{epoch_val_acc:.4f}",
            ])
        
        # ---- Save best checkpoint ----
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(
                "New best model at epoch %d with Val Acc: %.2f%% -> %s",
                epoch + 1, best_val_acc, best_ckpt_path
            )

        # Stream updates to GUI via callback
        if epoch_callback is not None:
            epoch_callback(epoch, history)

    final_ckpt_path = os.path.join(ckpt_dir, "final.pth")
    torch.save(model.state_dict(), final_ckpt_path)
    logger.info("Saved final model checkpoint to %s", final_ckpt_path)

    # ---- Test evaluation ----
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    history.test_acc = 100.0 * test_correct / test_total
    logger.info("Test Accuracy: %.2f%%", history.test_acc)



    return history
