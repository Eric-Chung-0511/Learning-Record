# 🏃‍♂️ Human Activity Recognition with Smartphones 📱

## 🔍 Project Overview:
* The Human Activity Recognition (HAR) database was created using data from 30 participants performing daily activities while wearing a Samsung Galaxy S II smartphone on their waist.

* The aim is to classify six types of activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying.

* The smartphone's built-in accelerometer and gyroscope recorded 3-axial linear acceleration and angular velocity at a rate of 50Hz. Data collection was supervised through video recordings to ensure accurate labeling.

* The data was preprocessed with noise filters and segmented into 2.56-second windows with 50% overlap, resulting in 128 readings per window.

* A Butterworth low-pass filter with a 0.3 Hz cutoff frequency was used to separate gravitational and body motion components of the acceleration signal. Feature vectors were extracted from each window, incorporating both time and frequency domain variables.

## ⚙️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* StandardScaler
* IQR (Handle Outliers)
* Label Encoder
* Label Binarize
* PCA
* FunctionTransformer
* ColumnTransformer
* Classification Report
* AUC_ROC Score
### 🖥️ Machine Learning:
* XGBoost 
* RandomForest
* SVM
* Logistic Regression
* AdaBoost

## 🌐 Skills Detail:
### 📊 Data Handling and Visualization:
* Python Libraries: Used Pandas for data manipulation and analysis, NumPy for numerical operations, and Matplotlib and Seaborn for creating  plots like barplot to visualize data distributions and inter-variable relationships. These visualizations help highlight key features and relationships critical to understanding fraud dynamics.

### ⚒️ Data Preprocessing:
* **Handling Outliers using IQR**: The Interquartile Range (IQR) is a statistical measure used to identify the spread of the middle 50% of data points in a dataset. It is calculated as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data. The IQR helps in identifying outliers by defining thresholds. **Typically, any data point that lies more than 1.5 times the IQR below Q1 or above Q3 is considered an outlier.** In this case, two techniques are applied: **setting outliers to the median or using quantile-based imputation**. The choice of method (median or quantile) can be customized based on the data distribution and analysis needs.

* **FunctionTransformer for Pipeline Integration**: To streamline the preprocessing steps within a pipeline, I employ **FunctionTransformer**. This tool allows us to wrap custom functions for use within the Pipeline class, ensuring a smooth integration of our preprocessing steps, such as outlier handling and scaling.

* **ColumnTransformer for Outliers and Scaling**: Create a preprocessor object, which includes a ColumnTransformer. The ColumnTransformer is used to apply different preprocessing techniques to different columns. In this case, it handles outliers and scales the data, ensuring each feature is processed appropriately.

* **Pipeline with PCA for Dimensionality Reduction**: Extend the preprocessing pipeline to include Principal Component Analysis (PCA) for dimensionality reduction. This step helps in reducing the number of features while retaining most of the variance, which is crucial for improving model performance and computational efficiency.

### 🤖 Model Building and Performance:
* **Model Evaluation with Cross-Validation**: Using cross-validation, we evaluate various models to determine which performs better when using the median or quantile-based outlier handling. This process involves splitting the data into multiple folds, training the model on each fold, and assessing performance metrics to find the best approach.
* **Model Performance Summary**
  
| Mean Train Accuracy | Mean Test Accuracy (CV) | Test Accuracy | Cross-Validation Errors | Model Name           | Method  |
|---------------------|-------------------------|---------------|-------------------------|----------------------|---------|
| 0.977045            | 0.968204                | 0.970088      | 0.031796                | SVC                  | quantile|
| 0.982316            | 0.968033                | 0.969409      | 0.031967                | Logistic Regression  | median  |
| 0.982231            | 0.967693                | 0.970088      | 0.032307                | Logistic Regression  | quantile|
| 0.976875            | 0.967183                | 0.966689      | 0.032817                | SVC                  | median  |
| 1.000000            | 0.951370                | 0.953093      | 0.048630                | XGBoost              | quantile|

* The models were evaluated without further hyperparameter tuning due to their already high accuracy. However, in typical scenarios, it is crucial to perform hyperparameter optimization to ensure the models achieve their best performance.

* **Classification Report and ROC Curve**: For the top 5 performing models, we generate and visualize the ROC curve and Classification Report. These metrics provide a detailed view of each model's performance, highlighting areas of strength and potential improvement.

* **To prevent overfitting, further evaluation is necessary. Techniques such as regularization, cross-validation on different subsets of data, and ensuring the model's performance on a completely separate test set are critical.**

* Overfitting occurs when a model learns the training data too well, including its noise and outliers, which negatively impacts its performance on new, unseen data. Therefore, continuous monitoring and validation against overfitting are essential steps in the model evaluation process.

* **Model Saving and Reloading for Testing**: Finally, the best-performing models are saved and later reloaded to test their accuracy on new, unseen data. This step ensures the robustness of our models and their ability to generalize well to new inputs.

## 📜 Conclusion:
* In this project, we tackled the task of human activity recognition using data collected from smartphones. We meticulously handled outliers using the IQR method with both median and quantile imputation techniques.

*  **Leveraging FunctionTransformer allowed us to seamlessly integrate custom preprocessing steps into a pipeline. We employed ColumnTransformer for efficient outlier handling and scaling, followed by dimensionality reduction using PCA.**

* Our model evaluation process, supported by cross-validation, ensured robust performance assessment across various models. We compared the effectiveness of different outlier handling methods and selected the top-performing models based on their cross-validation results.

* **To further ensure the reliability of our models, we visualized their ROC curves and generated classification reports, emphasizing the need to prevent overfitting through continuous evaluation.**

* By saving and reloading our models, we confirmed their ability to generalize well on new data, demonstrating their robustness. This comprehensive approach not only highlights the effectiveness of our preprocessing and evaluation techniques but also underscores the importance of rigorous model validation in achieving accurate and reliable human activity recognition.


## 📄 Viewing Jupyter Notebooks
* Sometimes there's bug on GitHub, if you encounter any problems displaying the Jupyter Notebooks directly on GitHub, you can view this project with the following link:
  [Human Activity Recognition with Smartphones](https://nbviewer.org/github/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Human%20Activity%20Recognition%20with%20Smartphones/Human%20Activity%20Recognition%20with%20Smartphones%20_Eric.ipynb)

  Thank you for your understanding!😊


  
