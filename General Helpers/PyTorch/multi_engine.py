"""
Contains functions for training and testing a PyTorch model with advanced metrics and features.
"""
from typing import Dict, List, Tuple

import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               metrics: Dict[str, torchmetrics.Metric]) -> Tuple[float, Dict[str, float]]:
    """Trains a PyTorch model for a single epoch with advanced metrics.

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    metrics: A dictionary of torchmetrics to compute during training.

    Returns:
    A tuple of training loss and a dictionary of metrics.
    """
    model.train()
    train_loss = 0
    metric_results = {name: 0 for name in metrics.keys()}

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(y_pred, dim=1)
        for name, metric in metrics.items():
            metric_results[name] += metric(y_pred_class, y).item()

    train_loss /= len(dataloader)
    metric_results = {name: total / len(dataloader) for name, total in metric_results.items()}
    return train_loss, metric_results

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              metrics: Dict[str, torchmetrics.Metric]) -> Tuple[float, Dict[str, float]]:
    """Tests a PyTorch model for a single epoch with advanced metrics.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    metrics: A dictionary of torchmetrics to compute during testing.

    Returns:
    A tuple of testing loss and a dictionary of metrics.
    """
    model.eval()
    test_loss = 0
    metric_results = {name: 0 for name in metrics.keys()}

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(y_pred, dim=1)
            for name, metric in metrics.items():
                metric_results[name] += metric(y_pred_class, y).item()

    test_loss /= len(dataloader)
    metric_results = {name: total / len(dataloader) for name, total in metric_results.items()}
    return test_loss, metric_results

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          early_stopping_patience: int = None) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model with advanced features like learning rate scheduling and early stopping.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    scheduler: A learning rate scheduler (optional).
    early_stopping_patience: Number of epochs with no improvement after which training will be stopped (optional).

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy, precision, recall, and f1-score metrics. Each metric has 
    a value in a list for each epoch.
    """
    results = {"train_loss": [],
               "train_acc": [],
               "train_precision": [],
               "train_recall": [],
               "train_f1": [],
               "test_loss": [],
               "test_acc": [],
               "test_precision": [],
               "test_recall": [],
               "test_f1": []}

    metrics = {
        "accuracy": Accuracy().to(device),
        "precision": Precision().to(device),
        "recall": Recall().to(device),
        "f1": F1Score().to(device)
    }

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in tqdm(range(epochs)):
        train_loss, train_metrics = train_step(model, train_dataloader, loss_fn, optimizer, device, metrics)
        test_loss, test_metrics = test_step(model, test_dataloader, loss_fn, device, metrics)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_metrics['accuracy']:.4f} | "
            f"train_precision: {train_metrics['precision']:.4f} | "
            f"train_recall: {train_metrics['recall']:.4f} | "
            f"train_f1: {train_metrics['f1']:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_metrics['accuracy']:.4f} | "
            f"test_precision: {test_metrics['precision']:.4f} | "
            f"test_recall: {test_metrics['recall']:.4f} | "
            f"test_f1: {test_metrics['f1']:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_metrics['accuracy'])
        results["train_precision"].append(train_metrics['precision'])
        results["train_recall"].append(train_metrics['recall'])
        results["train_f1"].append(train_metrics['f1'])
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_metrics['accuracy'])
        results["test_precision"].append(test_metrics['precision'])
        results["test_recall"].append(test_metrics['recall'])
        results["test_f1"].append(test_metrics['f1'])

        if scheduler:
            scheduler.step(test_loss)

        if early_stopping_patience:
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return results
