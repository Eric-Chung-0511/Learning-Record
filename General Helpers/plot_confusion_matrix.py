# classes: list, default=None, Class names. If None, numeric labels are used.
def plot_confusion_matrix(y_true, y_pred, figsize=(22, 22), classes=None, text_size=10):
    """
    Plots a confusion matrix and its normalized version.

    Parameters:
    y_true: array-like, shape (n_samples,)
        True labels.
    y_pred: array-like, shape (n_samples,)
        Predicted labels.
    figsize: tuple, default=(10, 10)
        Size of the plot.
    classes: list, default=None
        Class names. If None, numeric labels are used.

    Returns:
    None
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix (prevent division by zero)
    if cm.shape[0] > 1:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_norm = cm.astype('float') / cm.sum()

    # Number of classes
    n_classes = cm.shape[0]

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=figsize)

    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)

    # Add a color bar
    fig.colorbar(cax)

    # Set class labels
    if classes:
        labels = classes
    else:
        labels = np.arange(n_classes)

    ax.set(title='Confusion Matrix', xlabel='Predicted Label', ylabel='True Label',
           xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=labels, yticklabels=labels)

    # Move the X-axis label to the bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2

    # Plot text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)', horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black', size=15)

    # Show the plot
    plt.show()
