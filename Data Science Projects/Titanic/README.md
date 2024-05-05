# 🚢 Titanic Survival Project 🌊 

## 🔍 Project Overview:
* This project aims to analyze the historical Titanic dataset to understand the factors that influenced survival rates during the tragic event. 
* By applying machine learning techniques, we seek to predict survival outcomes based on various passenger characteristics, thereby gaining deeper insights into this significant historical incident.

## 🛠️ Skills Used:
### 🔧 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* Standard Scaler
* OneHot Encoder
* Label Encoder
* Frequecy Encoder
* Mean Encoder
* Pipeline
* Cross Validation
### ⚙️ Machine Learning:
* RandomForest
* Gradient Boosting
* SVM
* KNN
* GridSearchCV

## 📊 Skills Detail:
### 👀 Data Handling and Visualization:
* Leveraged Pandas for data manipulation and NumPy for numerical operations. Utilized Matplotlib and Seaborn for visualizations, including histograms, skewness and more to explore feature distributions and relationships.

### 🔢 Data Preprocessing:  
- **Outlier Detection**: Identified and examined outliers using histograms. The Interquartile Range (IQR) method was used to quantify variability and filter out extreme values.
  This method involves calculating the IQR as the difference between the 75th and 25th percentiles and is highly effective in robustly identifying outliers in skewed data.

### 🔨 Feature Engineering:
- **Simplification of Nominal Data**: Reduced complexity in categorical data by removing extraneous titles (e.g., 'Sir', 'Lady') and synthesized features like 'Siblings' and 'Spouse' into a single 'Family' feature to better represent family size.
  
- **Encoding Skills**:
   - **One-Hot Encoder**: Transformed categorical variables into a form that could be provided to ML algorithms to do a better job in prediction.
     
   - **Label Encoder**: Assigned a unique integer based on order of appearance to categorical labels.
     
   - **Mean Encoder**: Replaced categories with the mean value of the target variable, providing an average effect of the category.
     
   - **Frequency Encoder**: Converted categories into their frequencies, simplifying the input for algorithms.

### ⚙️ Machine Learning:
- **Parameter Optimization**: Utilized GridSearchCV to methodically search for the optimal parameters for each model. This approach systematically tests a range of parameter combinations, using cross-validation to ensure the selection enhances the model's ability to generalize to new data.

- **Model Application**: Employed various models such as RandomForest, Gradient Boosting, KNN, SVM to harness different statistical strengths. Each model was trained using data prepared with different encoders to evaluate the effect of these preprocessing strategies on predictive performance.

### 📈  Model Assessment:
- **Evaluation Metrics**: The performance of various machine learning models was evaluated using the mean test score from cv_results_, which reflects the average accuracy across all cross-validation folds. This measure is critical for assessing how well a model generalizes to unseen data and for comparing the effectiveness of different model configurations.

## 🎯 Conclusion:
* Through meticulous data handling, preprocessing, and model evaluation, this project elucidated the key factors that influenced survival on the Titanic.
* It demonstrated the effectiveness of combining feature engineering with sophisticated machine learning models to make predictions that are not only accurate but also interpretable and meaningful in understanding historical events.

## 📚 Acknowledgments and References:
* I am grateful to the developers and researchers whose work has significantly influenced this project. Below are the resources that have been instrumental:

* https://www.kaggle.com/code/volhaleusha/titanic-tutorial-encoding-feature-eng-81-8#Part-8:-Conclusion
