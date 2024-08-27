"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data with advanced augmentations.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose, 
        batch_size: int,
        num_workers: int = NUM_WORKERS,
        use_augmentation: bool = False,
        aug_transform: transforms.Compose = None
):
    """Create training and testing DataLoaders with optional data augmentation.

    Takes in training and testing directory paths and applies specified 
    torchvision transforms. Optionally applies augmentation on the training 
    data for better model generalization.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        train_transform: Basic torchvision transforms to perform on training data.
        test_transform: Basic torchvision transforms to perform on testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        use_augmentation: A boolean to apply augmentation transforms to training data.
        aug_transform: Additional torchvision transforms for augmentation.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """
    # Apply augmentation if specified
    if use_augmentation and aug_transform:
        train_transform = transforms.Compose([aug_transform, train_transform])

    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into DataLoaders
    train_dataloader = DataLoader(train_data, 
                                  batch_size=batch_size, 
                                  num_workers=num_workers, 
                                  shuffle=True,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False,
                                 pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
