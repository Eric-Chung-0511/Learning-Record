import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Print the AUC value
    print(f'AUC: {roc_auc:.2f}')

# Example usage:
# plot_roc_curve(svc, X_test_pca, y_test)
