import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# plot fot binary
def plot_roc_curve(model, X_test, y_test):
    """
    Plot ROC curve for a given model and test data.

    Parameters:
    model (object): Trained classifier with a predict_proba method.
    X_test (array-like): Test features.
    y_test (array-like): True labels for the test set.

    Returns:
    None
    """
    # Get the predicted probabilities for the positive class
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Compute the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(10, 8)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

    # Print the AUC value
    print(f'AUC: {roc_auc:.2f}')

# Example usage:
# plot_roc_curve(svc, X_test_pca, y_test)


# Plot for multiclasses
def plot_roc_curve_multiclass(model, X_test, y_test, n_classes=3):
   
    # Get the predicted probabilities for each class
    y_pred_proba = model.predict_proba(X_test)

    # Compute the log loss
    logloss = log_loss(y_test, y_pred_proba)
    print("Log Loss:", logloss)

    # Initialize dictionaries to store false positive rates, true positive rates, and ROC AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curve
    plt.figure(figsize=(7, 5))
    colors = ['blue', 'green', 'red']  # Change colors as needed
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
