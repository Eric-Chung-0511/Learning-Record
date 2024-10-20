# 💳 Credit Card Fraud Detection 🔍

## ✨ Project Overview:
* It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

* This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

## ⚙️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* StandardScaler
* IQR (Handle Outliers)
* Adasyn
* BayesSearchCV
* XGBoost
* classification Report
* ROC Curve

## 🤖 Skills Detail:
### 👓 Data Handling and Visualization:
* Python Libraries: Used Pandas for data manipulation and analysis, NumPy for numerical operations, and Matplotlib and Seaborn for creating diverse plots like histograms, box plots and more  to visualize data distributions and inter-variable relationships. These visualizations help highlight key features and relationships critical to understanding fraud dynamics.

### ⚒️ Data Preprocessing:
* **Log Transformation on 'Amount'**: To address skewed distributions typically seen with transaction amounts in fraud data, applied a logarithmic transformation to the 'Amount' feature. This helps normalize the data, reducing the influence of extreme values on the analysis.
  
* **Innovative Outlier Handling with IQR**: The Interquartile Range (IQR) is a statistical measure used to identify the spread of the middle 50% of data points in a dataset. It is calculated as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data. The IQR helps in identifying outliers by defining thresholds. **Typically, any data point that lies more than 1.5 times the IQR below Q1 or above Q3 is considered an outlier.**
  
* **Instead of simply removing outliers, implemented a method to handle them by replacing with the median or capping based on quantiles.** This approach, crucial in imbalanced datasets like fraud detection, helps maintain essential data points and avoids bias towards the majority class.

* **Data Splitting**: Segregated the data into training, validation, and test datasets to ensure a thorough evaluation phase.

* **Feature Scaling with StandardScaler**: Utilized StandardScaler to normalize the features within the dataset, ensuring that each feature contributes equally to the analysis and subsequent model predictions. This is crucial in algorithms that are sensitive to the scale of input data, like most machine learning algorithms, because it enhances their performance by providing a level playing field.

* **Balancing data with ADASYN**:
  - **Principle of ADSYN**: ADASYN (Adaptive Synthetic Sampling) is a technique used to create synthetic samples for the minority class in an imbalanced dataset. The fundamental idea behind ADASYN is to use a weighted distribution for different minority class samples according to their level of difficulty in learning, where more synthetic data is generated for harder samples. Specifically, it focuses on generating new samples next to the existing minority samples that are wrongly classified using a k-nearest neighbors algorithm.

  - **Differences Between ADASYN and SMOTE**:
    - **Sample Generation Focus**: While both ADASYN and SMOTE generate synthetic samples by interpolating between existing minority class samples, ADASYN places more emphasis on those samples that are difficult to classify. SMOTE generates the same number of samples for each minority class instance, treating all instances equally.
      
    - **Adaptive Nature**: ADASYN adapts to the inherent data structure. It uses a density distribution to automatically decide the number of synthetic samples needed for each minority sample, particularly focusing on the regions where the classifier has performed poorly, such as where the samples are misclassified.
    
    - **Handling of Borderline Examples**: ADASYN tends to focus more on the borderline examples, making it better suited for dealing with problems where the classes are not only imbalanced but also have complex decision boundaries.
  - **Why Use ADASYN**:
    - **Targeting Difficult Cases**: ADASYN is particularly effective in generating synthetic samples for those cases that are difficult to learn, which is crucial in complex scenarios like fraud detection where fraudulent behaviors are diverse and hard to predict.
      
    - **Boosting Minority Class Performance**: By focusing on the harder-to-classify, near-boundary samples, ADASYN improves classification accuracy for the minority class, which is essential in areas such as fraud detection where detecting rare events accurately is critical.
      
    - **Adapting to Complex Data Structures**: ADASYN excels in scenarios with complex decision boundaries and high class imbalance, thanks to its ability to adaptively generate more samples where they are most needed, enhancing the model's overall robustness and effectiveness.
   
### 🔝 XGBoost Classifier:
* **Structure**: XGBoost is a powerful and efficient implementation of gradient boosting. It works by building an ensemble of decision trees, where each new tree corrects the errors made by the previous ones. This iterative process helps the model learn complex patterns in the data and improve prediction accuracy, particularly on challenging datasets like fraud detection.

* **Boosting Mechanism**: In XGBoost, each new tree is added to the model to correct the residuals (errors) from the previous trees. The final model is a weighted sum of all the trees, resulting in a strong predictive model that excels in handling both large and imbalanced datasets, which are common in fraud detection.
  
* **Evaluation and Regularization**: XGBoost incorporates regularization techniques to prevent overfitting, such as L1 and L2 regularization. It also allows early stopping, which halts the training process when there is no further improvement on the validation set, ensuring the model doesn't overfit.
  
* **Bayesian Optimization**: Instead of manually tuning the hyperparameters, Skopt's Bayesian Optimization is used to automatically find the best configuration. This approach constructs a probabilistic model to predict the best hyperparameters and iteratively improves the performance of the model by efficiently searching through the parameter space. 

### 🧭 Model Evaluation:
* **Precision-Recall Curve Analysis**: The Receiver Operating Characteristic (ROC) curve provides a graphical representation of the true positive rate (recall) against the false positive rate. The AUC (Area Under the Curve) score summarizes the ROC curve by providing a single value that reflects the model’s ability to distinguish between positive and negative classes. In this project, a high **AUC score (0.99)** signifies the model's excellent discriminative power in detecting fraudulent transactions.
  
* **Phased Testing Approach**: Validated the model on the validation set followed by the test set, using these separate datasets to gauge the model’s ability to generalize to new, unseen data.
  
* **Comprehensive Performance Metrics**: Classification reports were generated to provide a detailed breakdown of the model's performance across precision, recall, F1-score, and accuracy. These metrics were used to evaluate both the validation and test sets, ensuring that the model generalizes well to unseen data.

## 🎯 Conclusion:
* This project highlighted the importance of using advanced machine learning techniques like **XGBoost and Bayesian Optimization** to handle complex and imbalanced datasets. XGBoost's gradient boosting mechanism, coupled with Bayesian hyperparameter optimization, significantly improved the model's performance in detecting fraudulent transactions.

* In the context of credit card fraud detection, **achieving a high AUC score and recall is critical. The model's ability to distinguish between legitimate and fraudulent transactions with a 99% AUC demonstrates its reliability in real world scenarios.** Focusing on high recall ensures that fewer fraudulent transactions are missed, which is vital in protecting financial institutions from significant financial losses.
  
## 📚 Acknowledgments and References:
* This project was influenced by several resources and contributions from the data science community. Special thanks to the Kaggle community and the following reference:

* https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 📄 Viewing Jupyter Notebooks
* Sometimes there's bug on GitHub, if you encounter any problems displaying the Jupyter Notebooks directly on GitHub, you can view this project with the following link:
  [Credit Card Fraud Detection](https://nbviewer.org/github/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Credit%20Card%20Fraud%20Detection/Credit%20Card%20Fraud%20Detection_Eric.ipynb)

  Thank you for your understanding!😊
