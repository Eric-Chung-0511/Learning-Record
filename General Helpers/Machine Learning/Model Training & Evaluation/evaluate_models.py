import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

def evaluate_models(X_train, y_train, classifiers=None):
    """
    Evaluate multiple classifiers using cross-validation and return a DataFrame with the results.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    classifiers (list, optional): List of tuples with classifier name and instance. 
                                  If None, a default list of classifiers will be used.

    Returns:
    pd.DataFrame: DataFrame containing the evaluation results for each classifier.
    """
    if classifiers is None:
        # Default list of classifiers
        classifiers = [
            ("Logistic Regression", LogisticRegression(max_iter=1500, random_state=42, n_jobs=-1)),
            ("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ('SVC', SVC(random_state=42)),
            ('Decision Tree', DecisionTreeClassifier(random_state=42)),
            ("Random Forest", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ("AdaBoost", AdaBoostClassifier(random_state=42)),
            ("XGBoost", XGBClassifier(random_state=42, n_jobs=-1))
        ]
    
    results = []
    
    for model_name, model in classifiers:
        try:
            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # Perform cross-validation
            cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1,
                                        return_train_score=True)
            
            # Calculate cross-validation error
            cross_val_error = 1 - np.mean(cv_results['test_score'])
            
            # Append results
            results.append({
                'Model Name': model_name,
                'Mean Train Accuracy': np.mean(cv_results['train_score']),
                'Mean Test Accuracy': np.mean(cv_results['test_score']),
                'Cross-Validation Errors': cross_val_error
            })
        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
    
    # Create a DataFrame with the results
    result_df = pd.DataFrame(results)
    
    # Sorting by Mean Test Accuracy in descending order
    result_df = result_df.sort_values(by='Mean Test Accuracy', ascending=False).reset_index(drop=True)
    
    return result_df

# Example usage:
# result_df = evaluate_models(X_train_pca, y_train)
# display(result_df)
