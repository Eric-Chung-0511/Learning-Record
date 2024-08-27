"""
Contains functionality for creating PyTorch DataLoaders from
prepared datasets, with optional data augmentation.
"""

from torch.utils.data import DataLoader
from torchvision import transforms
import os

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
        train_data, 
        test_data,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS,
        use_augmentation: bool = False,
        aug_transform: transforms.Compose = None
):
    """Create training and testing DataLoaders with optional data augmentation.

    Takes in prepared training and testing datasets and applies the provided transforms.
    Optionally applies augmentation on the training data for better model generalization.

    Args:
        train_data: A torchvision dataset instance for training data.
        test_data: A torchvision dataset instance for testing data.
        transform: Basic torchvision transforms to perform on both training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.
        use_augmentation: A boolean to apply augmentation transforms to training data.
        aug_transform: Additional torchvision transforms for augmentation.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    """
    # Apply basic transform to datasets
    train_data.transform = transform
    test_data.transform = transform
    
    # Apply augmentation if specified
    if use_augmentation and aug_transform:
        train_data.transform = transforms.Compose([aug_transform, transform])

    # Get class names
    class_names = train_data.classes

    # Create DataLoaders
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
