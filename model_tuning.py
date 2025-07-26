import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import logging

# ---------------- Logging ----------------
logging.basicConfig(
    filename='process.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------- Utils ----------------
def load_params(path: str = "params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- Model ----------------
class CustomCNN(nn.Module):
    def __init__(self, channels, kernel_sizes, num_classes=10):
        """
        channels: list[int]  -> out_channels for each conv block
        kernel_sizes: list[int] (same length as channels)
        """
        super().__init__()
        assert len(channels) == len(kernel_sizes), \
            "channels and kernel_sizes must have the same length"

        layers = []
        in_channels = 3
        for out_ch, k in zip(channels, kernel_sizes):
            layers.append(nn.Conv2d(in_channels, out_ch, kernel_size=k, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_ch
        self.conv = nn.Sequential(*layers)

        # compute spatial size after all pools: 32 // (2 ** L)
        L = len(channels)
        spatial = 32 // (2 ** L)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * spatial * spatial, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


if __name__ == "__main__":
    # ------------- Load params -------------
    params = load_params()

    seed = params['version']['seed']
    lr = params['parameter']['lr']
    batch_size = params['parameter']['batch_size']
    num_epochs = params['parameter']['num_epochs']

    # Treat conv_layers in params as the list of output channels (your YAML shows that)
    conv_channels = params['parameter']['conv_layers']
    # kernel_sizes can be a single int or a list
    ks = params['parameter']['kernel_sizes']
    if isinstance(ks, int):
        kernel_sizes = [ks] * len(conv_channels)
    else:
        kernel_sizes = ks

    # Paths (assuming you split to output_data/train, /val, /test)
    data_root = params['version']['output_data']
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')  # you used 'test' before; keep val here

    # ------------- Reproducibility -------------
    set_seeds(seed)

    # ------------- Data -------------
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_ds = ImageFolder(train_dir, transform=transform)
    val_ds = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(train_ds.classes)
    logger.info("Starting training")
    logger.info(f"Classes: {train_ds.classes}")

    # ------------- Model / Optim -------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(conv_channels, kernel_sizes, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------- Training loop -------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / total


        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100.0 * val_correct / val_total

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

    # ------------- Save -------------
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    logger.info("Model saved to models/model.pth")