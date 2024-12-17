import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import logging

class LatentSpaceNet(nn.Module):
    def __init__(self, latent_dim=256):
        super(LatentSpaceNet, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Latent space
        self.fc_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        latent = self.fc_latent(x)
        output = self.classifier(latent)
        return output, latent

def train_and_extract_latents(batch_size=128, latent_dim=256, epochs=10, save_path='latent_space.npy'):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = LatentSpaceNet(latent_dim=latent_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    logger.info("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                logger.info(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    logger.info("Training finished. Extracting latent representations...")

    # Extract latent representations
    model.eval()
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            _, latents = model(images)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate all latents and save
    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Save latent space representations
    np.save(save_path, latents)
    logger.info(f"Latent representations saved to {save_path}")
    
    # Return both latents and labels for further analysis
    return latents, labels

def analyze_latent_structure(latents, labels):
    """Analyze the structure of the latent space"""
    logger = logging.getLogger(__name__)
    
    # Basic statistics
    logger.info(f"Latent space shape: {latents.shape}")
    logger.info(f"Mean: {np.mean(latents):.3f}")
    logger.info(f"Std: {np.std(latents):.3f}")
    
    # Per-class statistics
    for i in range(10):  # CIFAR-10 has 10 classes
        class_latents = latents[labels == i]
        logger.info(f"Class {i}:")
        logger.info(f"  Mean: {np.mean(class_latents):.3f}")
        logger.info(f"  Std: {np.std(class_latents):.3f}")
        logger.info(f"  Number of samples: {len(class_latents)}")

if __name__ == "__main__":
    # Train model and extract latents
    latents, labels = train_and_extract_latents(
        batch_size=128,
        latent_dim=256,
        epochs=10,
        save_path='cifar10_latents.npy'
    )
    
    # Analyze latent space structure
    analyze_latent_structure(latents, labels)