from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def objection(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    """
    Objective function for Bayesian optimization to optimize RandomForestClassifier.

    Parameters:
    n_estimators (int): Number of trees in the forest.
    max_depth (int): Maximum depth of the tree.
    min_samples_split (int): Minimum number of samples required to split an internal node.
    min_samples_leaf (int): Minimum number of samples required to be at a leaf node.

    Returns:
    float: Mean accuracy score from cross-validation.
    """
    # Convert parameters to integers
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    
    # Initialize the RandomForestClassifier with given parameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                   max_features='sqrt', random_state=42)
    
    # Create a pipeline with SMOTE and the model
    rfc_pipeline = ImbPipeline([('smote', SMOTE(random_state=42)), ('model', model)])
    
    # Perform cross-validation and calculate the mean accuracy score
    score = cross_val_score(rfc_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    return np.mean(score)

# Define the parameter bounds for Bayesian optimization
pbounds = {'n_estimators': (100, 1000), 'max_depth': (10, 30), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 20)}

# Initialize the Bayesian optimizer
optimizer = BayesianOptimization(f=objection, pbounds=pbounds, random_state=42)

# Maximize the objective function
optimizer.maximize(init_points=20, n_iter=40)

# Print the best parameters and score
print(optimizer.max)

# Extract the best parameters
best_params = {'n_estimators': int(optimizer.max['params']['n_estimators']),
               'max_depth': int(optimizer.max['params']['max_depth']),
               'min_samples_split': int(optimizer.max['params']['min_samples_split']),
               'min_samples_leaf': int(optimizer.max['params']['min_samples_leaf']),
               'max_features': 'sqrt'}

# Initialize the model with the best parameters
model = RandomForestClassifier(**best_params, random_state=42)

# Create a pipeline with SMOTE and the model
rfc_pipeline = ImbPipeline([('smote', SMOTE(random_state=42)), ('model', model)])

# Fit the pipeline to the training data
rfc_pipeline.fit(X_train, y_train)

# Make predictions on the test data
rfc_prediction = rfc_pipeline.predict(X_test)

# Print training and accuracy scores
print(f'RandomForest Training Score is: {rfc_pipeline.score(X_train, y_train) * 100:.2f}%')
print(f'RandomForest Accuracy Score is: {accuracy_score(y_test, rfc_prediction) * 100:.2f}%')

# Print the classification report
print(classification_report(y_test, rfc_prediction))
