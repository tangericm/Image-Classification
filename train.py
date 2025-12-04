import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import *
from utils import *


def set_seed(seed=42):
    """Sets the seed for reproducibility for PyTorch and NumPy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#### TRAINING ####
if __name__ == "__main__":
    set_seed(42)
    
    # Hyperparameters
    N = 0 # Number of images used for training/validation (0 means all)
    num_classes = 10
    input_shape = (3, 64, 64)
    num_epochs = 50
    batch_size = 256
    learning_rate = 0.0001

    print("\n" + "="*50 + "\n" + "Loading CIFAR-10 Data")
    
    train, validation, test = load_CIFAR10(N, input_shape)

    print(f"Loaded {len(train)} images for training and {len(validation)} images for validation\n" + "="*50)

    print("\n" + "="*50 + "\nTraining on CIFAR-10:\n" + "="*50)

    # model = AlexNet(input_shape=input_shape, num_classes=num_classes)
    # model = VGG16(input_shape=input_shape, num_classes=num_classes)
    model = VGG19(input_shape=input_shape, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Using device: {device}\n")

    # Setup logging after model is created (so we can include model name in run)
    run_name = model.__class__.__name__
    logger, metrics_csv, ckpt_dir = setup_logging(run_name)
    logger.info("Starting training run: %s", run_name)
    # Log hyperparameters
    hparams = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "seed": 42,
        "device": str(device),
    }
    logger.info("Hyperparameters: %s", hparams)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                logger.info("Epoch [%d/%d], Batch [%d/%d], Loss: %.4f", epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item())
        
        train_accuracy = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        
        logger.info("Epoch [%d/%d] - Train Loss: %.4f, Train Acc: %.2f%% - Val Loss: %.4f, Val Acc: %.2f%%", epoch+1, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy)
        # Append metrics to CSV
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.6f}", f"{train_accuracy:.4f}", f"{val_loss:.6f}", f"{val_accuracy:.4f}"])

    logger.info("Training complete!")
    # Save the final model checkpoint
    final_ckpt = os.path.join(ckpt_dir, f"{model.__class__.__name__}_final.pth")
    torch.save(model.state_dict(), final_ckpt)
    logger.info("Model saved to %s", final_ckpt)