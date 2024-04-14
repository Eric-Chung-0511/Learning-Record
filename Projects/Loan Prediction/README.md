# 💵 Loan Prediction 💳

## 🔍 Project Overview:
* This project is about the company seeks to automate (in real time) the loan qualifying procedure based on information given by customers while filling out an online application form.
  
* It is expected that the development of ML models that can help the company predict loan approval in accelerating decision-making process for determining whether an applicant is eligible for a loan or not.

## 🛠️ Skills Used:
### Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* MinMaxScaler
* ImbPipeline
* SMOTE
* PyOD
* Cross Validation
* ROC Curve
* Classification Report
### Machine Learning:
* XGBoost 
* RandomForest
* SVM
* Logistic Regression
* AdaBoost
* Stakcing
* Voting
* Bayesian Optimization
* SMOTE

## 📊 Skills Detail:
### 👓 Data Handling and Visualization:
* Python Libraries: Used Pandas for data manipulation and analysis, NumPy for numerical operations, and Matplotlib and Seaborn for creating diverse plots like histograms, box plots and more  to visualize data distributions and inter-variable relationships.

### 🪛 Data Preprocessing:  
 - **Implementation with the PyOD Library**: For outlier detection, the PyOD library was utilized, providing access to several powerful outlier detection algorithms. Specifically, methods like Isolation Forest (IForest), K-Nearest Neighbors (KNN), Elliptical Envelope (ECOD), Local Outlier Factor (LOF), and Principal Component Analysis (PCA) were employed.
   
  - **Each of these algorithms offers a unique approach to identifying anomalies**:
    - **IForest**: Isolates anomalies instead of profiling normal data points, which is efficient for handling high-dimensional datasets.
      
    - **KNN**: Detects outliers by measuring the distance from a point to its neighbors, identifying those points that have a significantly longer distance.
     
    - **ECOD**: Assumes normal data points follow an elliptical distribution, effectively identifying outliers that deviate from this fit.
      
    - **LOF**: Measures local deviations of density compared to neighbors, which is effective in identifying outliers in a clustered dataset.
      
    - **PCA**: Reduces the dimensionality of the data, highlighting anomalies as those points that have large variations in the reduced dimensions.

- **Handling Data Imbalance**: Employed SMOTE to synthetically balance the class distribution in the dataset, enhancing the predictive performance of classifiers.

### ⚙️ Feature Engineering:
- **Imbalanced Data Pipeline**: Leveraged ImbPipeline to integrate resampling techniques directly within the modeling pipeline, ensuring a streamlined process from data preprocessing to model training.

### 🤖 Machine Learning:
- **Model Optimization**: Used Bayesian Optimization to systematically search for the optimal set of hyperparameters, thus improving model efficacy.
- **Model Evaluation and Selection**: Assessed various models through metrics such as classification reports and ROC curves to select the best performer.
- **Ensemble Techniques**: Advanced model accuracy using ensemble strategies like voting and stacking, which combine predictions from several models to improve overall accuracy.

### 🧭 Model Assessment:
- **Performance Metrics**: Utilized classification reports and ROC curves to evaluate the effectiveness and reliability of the predictive models. These reports provide detailed metrics including:
 - **Accuracy**: Measures the overall correctness of the model; the proportion of true results (both true positives and true negatives) among the total number of cases examined.
 - **Precision**: Indicates the ratio of true positives to all positives, reflecting the model's ability to avoid false positives.
 - **Recall (Sensitivity)**: Assesses the model's ability to detect all relevant instances, representing the ratio of true positives to the sum of true positives and false negatives.
 - **AUC (Area Under Curve)**: Evaluates the model's ability to discriminate between classes and is used as a summary of the ROC curve.
   
## 🎯 Conclusion:
* In this Loan Prediction project, I tackled the challenge of selecting the best model from several closely performing options. I evaluated five machine learning models Logistic Regression, RandomForest, XGBoost, AdaBoost, and SVM even two ensemble methods, Voting and Stacking, with SVC as the final estimator.
  
* This comparative analysis taught me that while SVM topped in accuracy, evaluating models on multiple metrics like Recall and Precision is crucial, particularly with imbalanced datasets where recognizing minority classes and ensuring accurate predictions are equally important.

* I employed Bayesian Optimization to fine-tune each model’s parameters, which not only boosted their performance but also enhanced my understanding of each model’s sensitivity to different settings.
  
* This project highlighted that effective model building goes beyond employing advanced algorithms; it requires a deep understanding of the data and thoughtful consideration of multiple evaluation metrics.

* The decision on which model to use should align with the project's specific objectives. If maximizing accuracy is the goal, SVC is ideal. However, for projects where reducing the risk of missing risky loans is paramount, options with higher Recall should be considered.
  
* Additional factors such as model interpretability, computational efficiency, and practical viability must also be taken into account to ensure the selected model meets real world demands effectively.

* This experience has underscored the value of domain knowledge in applying machine learning to tackle complex real-world challenges efficiently.
