from pyod.models.knn import KNN
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.lof import LOF
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Function to detect outliers using multiple models from the PyOD library
def detect_outliers(X, contamination=0.1):
    # Dictionary to store different outlier detection models
    models = {
        'KNN': KNN(contamination=contamination),
        'ECOD': ECOD(contamination=contamination),
        'IForest': IForest(contamination=contamination),
        'PCA': PCA(contamination=contamination),
        'LOF': LOF(contamination=contamination)
    }
    
    outliers = {}  # Dictionary to store the detected outliers by each model
    for model_name, model in models.items():  # Iterate over each model
        model.fit(X)  # Fit the model to the data
        outliers[model_name] = model.labels_  # Store the outlier labels
    return outliers  # Return the dictionary of outlier labels

# Function to find the best threshold for contamination using cross-validation
def find_best_threshold(X, y, models, start=None, end=None, step=None, n_splits=None):
    # Set default values if None are provided
    start = 0.1 if start is None else start
    end = 0.5 if end is None else end
    step = 0.1 if step is None else step
    n_splits = 5 if n_splits is None else n_splits

    best_threshold = start  # Initialize the best threshold
    best_f1_score = -1  # Initialize the best F1 score
    
    thresholds = np.arange(start, end, step)  # Generate a range of thresholds
    skf = StratifiedKFold(n_splits=n_splits)  # Stratified k-fold cross-validator
    
    for threshold in thresholds:  # Iterate over each threshold
        temp_f1_scores = []  # List to store F1 scores for the current threshold
        for train_index, test_index in skf.split(X, y):  # Split the data into train and test sets
            
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:  
                X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Update contamination parameter for each model
            temp_models = {name: clf.set_params(contamination=threshold) for name, clf in models.items()}
            for model_name, model in temp_models.items():  # Iterate over each model
                model.fit(X_train)  # Fit the model to the training data
                y_pred = model.predict(X_test)  # Predict the outliers on the test data
                temp_f1_scores.append(f1_score(y_test, y_pred))  # Calculate and store the F1 score
                
        avg_f1_score = np.mean(temp_f1_scores)  # Calculate the average F1 score for the current threshold
        if avg_f1_score > best_f1_score:  # Check if the current F1 score is the best so far
            best_f1_score = avg_f1_score  # Update the best F1 score
            best_threshold = threshold  # Update the best threshold
            
    return best_threshold  # Return the best threshold

# Dictionary to store the models with initial contamination parameter
models = {
    'KNN': KNN(contamination=0.1),
    'ECOD': ECOD(contamination=0.1),
    'IForest': IForest(contamination=0.1),
    'PCA': PCA(contamination=0.1),
    'LOF': LOF(contamination=0.1)
}

# Find the best threshold using the defined function
best_threshold = find_best_threshold(X, y.values, models)

print(f'Best contamination threshold: {best_threshold}')  # Print the best threshold

# Function to combine outlier results from multiple models
def combined_outliers(outliers, strategy='all'):
    # Combine outliers using the specified strategy
    if strategy == 'all':  # Strategy 'all' is equivalent to 'union'
        return np.any(list(outliers.values()), axis=0)
    elif strategy == 'union':  # 'union' strategy: outliers detected by any model
        return np.any(list(outliers.values()), axis=0)
    elif strategy == 'intersection':  # 'intersection' strategy: outliers detected by all models
        return np.all(list(outliers.values()), axis=0)
