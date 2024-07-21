# 🏦 Credit Score Classification 📊

## ✨ Project Overview:
* You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.

* Given a person’s credit-related information, build a machine learning model that can classify the credit score.

## ⚙️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Seaborn
* Matplotlib
* OS
* Re
* Label Encoder
* SMOTE
* StandardScaler

### 📈 Modeling, Parameter Tuning, and Evaluation:
* Bayesian Optimization
* ROC Curve
* LightGBM
* XGBoost
* RandomForest
* ANN (Artificial Neural Network)
* Classification Metrics: Accuracy, Precision, Recall, F1-score

## 🤖 Skills Detail:
### 👓 Data Handling and Visualization:
* **Identifying Missing Values**: Initially detected which features had missing values and categorized them into numerical and categorical features.
  
* **Handling Missing Values**: Processed missing values by setting ranges (e.g., Age: 0-120), using regex to clean categorical features, and removing unnecessary symbols. Visualized job distribution with bar charts.
  
* **Feature Engineering**: Added new features and removed unnecessary ones. Used correlation and VIF for initial feature selection. Employed quantile (lower 0.1, upper 0.9) to filter outliers, followed by box plots and histograms to check distributions. Applied log transformation to filter features with high skewness. Finally, used feature importance to determine the most crucial features.

### ⚒️ Data Preprocessing:
* **Log Transformation on 'Amount'**: Applied log transformation to normalize skewed 'Amount' feature.
  
* **Innovative Outlier Handling with Quantile Method**: Instead of simply removing outliers using IQR, employed a custom method to remove the smallest 10% and the biggest 10% of data points, thereby retaining the middle 80% of the data, crucial for handling imbalanced datasets.
  
* **Feature Selection with Correlation and VIF**: 
  - **Correlation**: Used correlation matrices to identify and remove highly correlated features to prevent multicollinearity and enhance model performance.
    
  - **Variance Inflation Factor (VIF)**: Further evaluated features using VIF to ensure minimal multicollinearity, thus improving the reliability of the regression coefficients.
    
  - **Multicollinearity**: Multicollinearity occurs when two or more predictor variables in a statistical model are highly correlated, meaning that one can be linearly predicted from the others with a substantial degree of accuracy. This can lead to problems in estimating the coefficients of the model, as it becomes difficult to determine the individual effect of each predictor. Avoiding multicollinearity is crucial because it can inflate the variance of the coefficient estimates and make the model unstable and difficult to interpret.
    
* **Feature Scaling with StandardScaler**: Normalized features to ensure equal contribution.
  
* **Balancing Data with SMOTE**: Used SMOTE to handle class imbalance by generating synthetic samples for the minority class.

### 🧬 Model Building and Evaluation:
* **Bayesian Optimization for Parameter Tuning**: Applied Bayesian Optimization for initial parameter tuning of LightGBM, XGBoost, and RandomForest models.
  
* **Model Training and Fine-Tuning**: Further fine-tuned models and evaluated using accuracy, precision, recall, F1-score, and ROC_AUC.
  
* **ANN Implementation**: Built and evaluated an Artificial Neural Network with the same metrics.

## 🎯 Conclusion:
The project highlighted the importance of comprehensive preprocessing in handling imbalanced datasets and optimizing machine learning models for credit score classification. Key insights and learnings from this project include:

1. **Feature Importance**:
    ```plaintext
    Feature                         Importance
    Credit_Mix_Encoded              0.136037
    Credit_History_Age_Months       0.106051
    Outstanding_Debt                0.102662
    Delay_from_due_date             0.070184
    Interest_Rate                   0.069043
    Debt_Per_Account                0.066341
    Debt_to_Income_Ratio            0.053501
    Annual_Income                   0.052415
    Delayed_Payments_Per_Account    0.051063
    Age                             0.051015
    Monthly_Inhand_Salary           0.050329
    Payment_Behaviour_Encoded       0.046997
    Num_of_Delayed_Payment          0.043725
    Num_Credit_Inquiries            0.043693
    Occupation_Encoded              0.036368
    Payment_of_Min_Amount_Encoded   0.020576
    ```

2. **Insights and Learnings**:
    * **Credit Mix and History**: Features like `Credit_Mix_Encoded` and `Credit_History_Age_Months` were identified as the most important, indicating that a diverse mix of credit and a longer credit history are strong indicators of creditworthiness.
      
    * **Outstanding Debt**: The amount of outstanding debt plays a significant role, suggesting that lower outstanding debt is associated with better credit scores.
      
    * **Delay from Due Date**: Timeliness of payments (`Delay_from_due_date`) is crucial, highlighting the importance of paying bills on time.

3. **Potential Improvements**:
    * **Feature Engineering**: Further feature engineering could enhance model performance. For example, creating interaction features or using domain knowledge to derive new features.
      
    * **Advanced Models**: Exploring advanced models and techniques, such as ensemble methods or gradient boosting, could provide better classification accuracy.
      
    * **Real-World Application**: In the context of credit score classification, achieving high recall is critical as it ensures fewer misclassifications of low credit scores, which is essential for reducing financial risk for the company.

## 📚 Acknowledgments and References:
* This project was influenced by several resources and contributions from the data science community. Special thanks to the Kaggle community and the following reference:

* [Credit Score Classification](https://www.kaggle.com/competitions/credit-score-classification)
