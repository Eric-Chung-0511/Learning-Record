import torch
import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

class Trainer:
    """
    A class to encapsulate the training, testing, and evaluation of a PyTorch model.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be trained and evaluated.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
        loss_fn (torch.nn.Module): Loss function used to calculate the error.
        device (torch.device): The device (CPU or GPU) on which computations are performed.
        num_classes (int): Number of classes in the classification task.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler (default is None).
        early_stopping_patience (int, optional): Number of epochs with no improvement after which training will be stopped (default is None).
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_dataloader: torch.utils.data.DataLoader, 
                 test_dataloader: torch.utils.data.DataLoader, 
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 device: torch.device,
                 num_classes: int,  
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 early_stopping_patience: int = None):
        """
        Initializes the Trainer class with the given model, data loaders, optimizer, loss function, and other training parameters.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained and tested.
            train_dataloader (torch.utils.data.DataLoader): DataLoader instance for training data.
            test_dataloader (torch.utils.data.DataLoader): DataLoader instance for testing data.
            optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters.
            loss_fn (torch.nn.Module): Loss function to minimize during training.
            device (torch.device): The device to perform computation on (e.g., 'cuda' or 'cpu').
            num_classes (int): Number of classes in the classification task.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to adjust the learning rate during training (default is None).
            early_stopping_patience (int, optional): Number of epochs with no improvement after which training stops (default is None).
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

        # Initialize metrics using torchmetrics
        self.metrics = {
            "accuracy": Accuracy(task='multiclass', num_classes=num_classes).to(device),
            "precision": Precision(task='multiclass', num_classes=num_classes).to(device),
            "recall": Recall(task='multiclass', num_classes=num_classes).to(device),
            "f1": F1Score(task='multiclass', num_classes=num_classes).to(device)
        }

    def train_step(self) -> Tuple[float, Dict[str, float]]:
        """
        Trains the model for a single epoch using the training dataloader.

        Returns:
            Tuple[float, Dict[str, float]]: Returns the average training loss and a dictionary of computed metrics.
        """
        self.model.train()  # Set the model to training mode
        train_loss = 0
        metric_results = {name: 0 for name in self.metrics.keys()}

        # Iterate over batches of data
        for X, y in self.train_dataloader:
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate metrics for this batch
            y_pred_class = torch.argmax(y_pred, dim=1)
            for name, metric in self.metrics.items():
                metric_results[name] += metric(y_pred_class, y).item()

        # Average the loss and metrics over all batches
        train_loss /= len(self.train_dataloader)
        metric_results = {name: total / len(self.train_dataloader) for name, total in metric_results.items()}
        return train_loss, metric_results

    def test_step(self) -> Tuple[float, Dict[str, float]]:
        """
        Tests the model for a single epoch using the testing dataloader.

        Returns:
            Tuple[float, Dict[str, float]]: Returns the average testing loss and a dictionary of computed metrics.
        """
        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0
        metric_results = {name: 0 for name in self.metrics.keys()}

        # No gradients needed for evaluation
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                test_loss += loss.item()

                # Calculate metrics for this batch
                y_pred_class = torch.argmax(y_pred, dim=1)
                for name, metric in self.metrics.items():
                    metric_results[name] += metric(y_pred_class, y).item()

        # Average the loss and metrics over all batches
        test_loss /= len(self.test_dataloader)
        metric_results = {name: total / len(self.test_dataloader) for name, total in metric_results.items()}
        return test_loss, metric_results

    def train(self, epochs: int) -> Dict[str, List[float]]:
        """
        Trains the model over a specified number of epochs, with optional learning rate scheduling and early stopping.

        Args:
            epochs (int): Number of epochs to train the model for.

        Returns:
            Dict[str, List[float]]: A dictionary containing the training and testing loss and metrics for each epoch.
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

        best_loss = float('inf')
        epochs_without_improvement = 0

        # Iterate over each epoch
        for epoch in tqdm(range(epochs)):
            train_loss, train_metrics = self.train_step()
            test_loss, test_metrics = self.test_step()

            # Print the metrics for this epoch
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

            # Store the metrics in the results dictionary
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

            # Update the learning rate if scheduler is provided
            if self.scheduler:
                self.scheduler.step(test_loss)

            # Implement early stopping
            if self.early_stopping_patience:
                if test_loss < best_loss:
                    best_loss = test_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        return results


    def classification_report(self, class_names: List[str]) -> None:
        """
        Generates and prints a classification report based on the test dataset.

        Args:
            class_names (List[str]): List of class names for the classification report.
        """
        self.model.eval()  # Set the model to evaluation mode
        y_true = []
        y_pred = []

        # Perform inference on the test set
        with torch.inference_mode():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                
                # Store the true labels and the predicted labels
                y_true.extend(y.cpu().numpy())
                y_pred.extend(torch.argmax(preds, dim=1).cpu().numpy())

        # Generate and print the classification report
        print(classification_report(y_true, y_pred, target_names=class_names))

# Example Usage
# # Start the timer
# from timeit import default_timer as timer
# start_time = timer()

# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# trainer = Trainer(
#         model=model,
#         train_dataloader=train_dataloader,
#         test_dataloader=test_dataloader,
#         optimizer=optimizer,
#         loss_fn=loss_fn,
#         device=device,
#         num_classes=3,
#         scheduler=scheduler,
#         early_stopping_patience=5)

# # Train the model
# results = trainer.train(epochs=20)

# # Print the classification report
# trainer.classification_report(class_names=class_names)

# # End the timer and print out how long it took
# end_time = timer()
# print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
