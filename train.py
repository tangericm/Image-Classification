import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import AlexNet
from utils import load_CIFAR10

def set_seed(seed=42):
    """Sets the seed for reproducibility for PyTorch and NumPy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Total number of images used for training/validation
N = 0 # Full dataset
num_classes = 10
input_shape = (3, 64, 64)
num_epochs = 50
batch_size = 256
learning_rate = 0.0001

#### TRAINING ####
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Loading CIFAR-10 Data")
    
    train, validation, test = load_CIFAR10(N, input_shape)

    print(f"Loaded {len(train)} images for training and {len(validation)} images for validation")
    print("="*50)


    print("\n" + "="*50)
    print("Training AlexNet on CIFAR-10:")
    print("="*50)

    model = AlexNet(input_shape=input_shape, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loaders
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Using device: {device}\n")

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
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%\n")

    print("Training complete!")
    # Save the model
    torch.save(model.state_dict(), './models/alexnet_cifar10.pth')
    print("Model saved as 'alexnet_cifar10.pth'")