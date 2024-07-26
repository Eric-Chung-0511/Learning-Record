class ModelEvaluator:
    def __init__(self, model=None, is_ann=False):

        self.model = model
        self.is_ann = is_ann

    def calculate_results(self, y_true, y_pred):

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
            'f1_score': f'{model_f1_score}%'}

        return model_results

    def plot_roc_curve_multiclass(self, X_test, y_test, n_classes=3):

        if self.model is None:
            raise ValueError("Model is not provided")

        # Convert y_test to numpy array if it's a pandas Series
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()

        # Encode y_test to one-hot if the model is ANN
        if self.is_ann:
            encoder = OneHotEncoder()
            y_test = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()

        # Get the predicted probabilities for each class
        if self.is_ann:
            y_pred_proba = self.model.predict(X_test)
        else:
            y_pred_proba = self.model.predict_proba(X_test)  # using predict_proba for probabilities

        # Ensure the shape of y_pred_proba is (n_samples, n_classes)
        # if it's binary you only need the positive prob ([:, 1]), so do not need this line of code.
        # y_pred_proba[:, 0]: The probability that a sample is predicted to belong to class 0.(negative)
        # y_pred_proba[:, 1]: The probability that a sample is predicted to belong to class 1.(positive)
        # so if it's binary problem, do not need this line of code
        assert y_pred_proba.shape == (X_test.shape[0], n_classes), "Shape of y_pred_proba is incorrect"

        # Compute log loss if the model is ANN
        if self.is_ann:
            logloss = log_loss(y_test, y_pred_proba)
            print("Log Loss:", logloss)

        # Initialize dictionaries to store false positive rates, true positive rates, and ROC AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute ROC curve and ROC AUC for each class
        for i in range(n_classes):
            if self.is_ann:
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            else:
                fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot the ROC curve
        plt.figure(figsize=(7, 5))
        colors = ['blue', 'green', 'red']
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

    def plot_roc_curve_multiclass_hard_voting(self, voting_clf, X_test, y_test, n_classes=3):

        # Convert y_test to numpy array if it's a pandas Series
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()

        # Get each model in the voting classifier
        classifiers = {name: clf for name, clf in voting_clf.named_estimators_.items()}

        # Initialize dictionaries to store fpr, tpr, roc_auc
        fpr = {name: dict() for name in classifiers}
        tpr = {name: dict() for name in classifiers}
        roc_auc = {name: dict() for name in classifiers}

        for name, model in classifiers.items():
            # Get the predicted probabilities
            y_pred_proba = model.predict_proba(X_test)

            # Calculate roc and auc for each class
            for i in range(n_classes):
                fpr[name][i], tpr[name][i], _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
                roc_auc[name][i] = auc(fpr[name][i], tpr[name][i])

        # Plot the ROC curve
        plt.figure(figsize=(14, 10))
        colors = ['blue', 'green', 'red']
        for name in classifiers:
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[name][i], tpr[name][i], color=color, lw=2,
                         label=f'ROC curve of class {i} for {name} (area = {roc_auc[name][i]:0.2f})')

        plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) for Hard Voting')
        plt.legend(loc="lower right")
        plt.show()
