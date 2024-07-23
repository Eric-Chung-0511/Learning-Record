# 🎼 Classify Song Genres from Audio Data Model 🎵

## 🔍 Project Overview:
* In recent years, streaming services with extensive music catalogs have become the primary way people listen to music. However, the vast selection can be overwhelming for users searching for new tunes that fit their taste.

* To address this, streaming services have employed methods to categorize music for personalized recommendations. One such method analyzes the raw audio data of songs to score various metrics.
  
* Today, I'll use a dataset from The Echo Nest to classify songs into 'Hip-Hop' or 'Rock' categories without listening to any tracks ourselves.

## 🛠️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* StandardScaler
* Cross Validation
* ROC Curve
* Classification Report
* Confusion Matrix
* Label Encoder
* Principal components analysis(PCA)
  
### 🖥️ Machine Learning:
* RandomForest
* SVM
* KNN
* Logistic Regression
* AdaBoost
* GridSearchCV

## 📊 Skills Detail:
### 🔬 Data Handling and Visualization:
 - **Python Libraries**: Utilized Pandas for data manipulation, NumPy for numerical operations, and Matplotlib and Seaborn for creating visualizations to explore the data.
   
 - **Exploratory Data Analysis (EDA)**: Conducted initial analysis using heatmaps and barplots to examine correlations between different features within the dataset.

### 🦾 Data Preprocessing:
 - **Random Forest for Feature Importance**: Implemented a RandomForest classifier to fit the data. This model is used to assess and rank feature importance, identifying which variables significantly impact the classification outcome.
   
### 🧮 Feature Engineering:
 - **Label Encoding**: Transformed categorical labels into numeric formats to prepare data for machine learning models. This conversion ensures effective processing of categorical data by algorithms.
   
 - **Dimensionality Reduction with PCA**: Employed PCA to reduce the number of variables in the dataset while retaining the essential information. This step is critical in simplifying the model's complexity, which enhances performance by focusing on the most relevant features.

### 🕹️ Machine Learning:
 - **Initial Model Evaluation**: Used cross-validation results, including mean train accuracy, mean test accuracy, and cross-validation errors, to initially evaluate the performance of various models. This step helped identify models that were overfitting.

 - **Parameter Tuning with GridSearch**: For models that did not exhibit overfitting, GridSearchCV was employed to fine-tune parameters. This method systematically searches through multiple combinations of parameters to find the best setup that improves model performance.

### ⚖️ Model Assessment:
 - **Final Model Evaluation**: Assessed the final models using the classification report and confusion matrix. The classification report provides detailed insights into the precision, recall, and F1-score of the model, while the confusion matrix visually represents the accuracy of predictions against actual classifications, helping to validate the effectiveness and reliability of the model in classifying the data accurately.

## 🎤 Conclusion:
 * This project effectively applied machine learning to classify song genres, identifying and addressing overfitting through initial cross-validation.
 
 * Parameter optimization via GridSearchCV refined our models, with final evaluations confirming their accuracy and reliability. The process underscored the importance of careful model selection and tuning in achieving precise classification outcomes.

## 📚 Acknowledgments and References:
* This project was influenced by several resources and contributions from the data science community. Special thanks to the Kaggle community and the following reference:

* https://www.kaggle.com/datasets/aniruddhachoudhury/classify-song-genres-from-audio-data


## 📄 Viewing Jupyter Notebooks
* Sometimes there's bug on GitHub, if you encounter any problems displaying the Jupyter Notebooks directly on GitHub, you can view this project with the following link:
  [Classify Song Genres from Audio Data Model](https://nbviewer.org/github/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Music%20Genre/Music%20Classification_Eric.ipynb)

  Thank you for your understanding!😊
