from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
    """
    Calculate and format the evaluation metrics for a classification model.

    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1 score as percentages.
    """
    
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate model precision, recall, and F1 score, '_' stands for support
    model_precision, model_recall, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Round accuracy to two decimal places
    model_accuracy = round(model_accuracy * 100, 2)
    
    # Round precision to two decimal places and convert to percentage
    model_precision = round(model_precision * 100, 2)
    
    # Round recall to two decimal places and convert to percentage
    model_recall = round(model_recall * 100, 2)
    
    # Round F1 score to two decimal places and convert to percentage
    model_f1_score = round(model_f1_score * 100, 2)
    
    # Store the results in a dictionary with percentage format
    model_results = {
        'accuracy': f'{model_accuracy}%',
        'precision': f'{model_precision}%',
        'recall': f'{model_recall}%',
        'f1_score': f'{model_f1_score}%'
    }

    return model_results
