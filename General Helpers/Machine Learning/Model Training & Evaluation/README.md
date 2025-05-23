#  📚 Function Overview

### plot_confusion_matrix([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/General%20Helpers/Examples/plot_confusion_matrix_example.ipynb))
Visualize the customized confusion matrix to evaluate the performance of classification models. Helps in identifying misclassifications and understanding model accuracy.

### Bayesian Optimization([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Loan%20Prediction/Loan%20Prediction%20_Eric.ipynb))
- **This function consists of several functions and steps aimed at optimizing and training a RandomForestClassifier using Bayesian optimization and handling imbalanced data with SMOTE.**
- **objection**: Objective function for Bayesian optimization to optimize RandomForestClassifier parameters.
- **Bayesian Optimization**: Process to find the best hyperparameters for the RandomForestClassifier.
- **Pipeline Creation**: Create a pipeline with SMOTE and the optimized RandomForestClassifier for handling imbalanced datasets.

### find_optimal_k([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/E-Commerce/E-Commerce%20Cluster%20_Eric.ipynb))
- **This function consists of a single function aimed at finding the optimal number of clusters for KMeans clustering using the Elbow Method.**
- **find_optimal_k**: Use the Elbow Method to determine the optimal number of clusters by plotting the sum of squared errors (SSE) against the number of clusters.

### evaluate_models([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Music%20Genre/Music%20Classification_Eric.ipynb))
- **This function consists of one main function aimed at evaluating multiple classifiers using cross-validation.**
- **evaluate_models**: Evaluate multiple classifiers using stratified k-fold cross-validation and return a DataFrame containing the evaluation results, including mean train accuracy, mean test accuracy, and cross-validation errors.

### plot_roc_curve([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Loan%20Prediction/Loan%20Prediction%20_Eric.ipynb))
- **This function consists of one main function aimed at plotting the ROC curve for a given model and test data.**
- **plot_roc_curve**: Plot the ROC curve for a trained classifier, compute the AUC, and display the curve with the AUC value, providing a visual representation of the model's performance.

### calculate_results([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/SkimLit/SkimLit_Project_RoBERTa_02.ipynb))
- **The first function calculates and formats the evaluation metrics for a classification model. It computes the accuracy, precision, recall, and F1 score, rounds them to two decimal places, converts them to percentages, and returns them in a dictionary.**

  
### (OOP)Model_Evaluator([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Credit%20Score%20Classification/Credit_Score_Classification_Eric_Final00.ipynb))
- **The second function using Object-Oriented Programming (OOP) style that calculates accuracy, precision, recall, and F1-score. Additionally, the function should include the ability to plot the ROC curve. It should be designed for multiple classes and be easily adaptable for binary classification scenarios.**






