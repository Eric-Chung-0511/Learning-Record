# Data Science Project Work Flow

## 🎯 Description and Goal

This workflow outlines the steps for completing a data science project, starting from data collection to model evaluation. Based on my own project experience, it provides detailed procedures for handling missing values, data preprocessing, feature engineering, and model selection.

## 📋 Steps

1. **📊 Data Collection and Preparation**
    - Assume the necessary data has been collected and converted into a CSV file or whatever the file is.
    - Read the data into your IDE (Jupyter Notebook, VSCode, Spyder, PyCharm, etc.).

2. **🔍 Initial Data Exploration**
    - Inspect the data using methods like `head()`, `info()`, `isna().sum()`, `describe()`, etc.

3. **⚖️ Check for Data Imbalance**
    - Use `value_counts(normalize=True)` to calculate the proportion of each class if applicable.

4. **🔠 Identify Data Types**
    - Classify features as Numeric or Categorical.
      
    - Separate the Categorical and Numerical data.
      

   ```python

   # Separate categorical columns
   categorical_columns = df.select_dtypes(include=['object', 'category']).columns

   # Separate numerical columns
   numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

   print(f'Categorical Columns:{list(categorical_columns)}')
   print(f'Numerical Columns:{list(numerical_columns)}')
   ```

5. **👀 Dive into Data and Visualize it**
   - Perform exploratory data analysis (EDA) to uncover patterns, spot anomalies, and test hypotheses.

   - Utilize visualizations to understand data distributions and relationships between variables.
  
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   # Scatter plot to visualize relationships between two numerical variables
   sns.scatterplot(data=df, x='feature1', y='feature2', hue='target')
   plt.title('Scatter Plot of Feature1 vs Feature2')
   plt.show()

   # Box plot to visualize the distribution of numerical data and identify outliers
   sns.boxplot(data=df, x='categorical_feature', y='numerical_feature')
   plt.title('Box Plot of Numerical Feature by Categorical Feature')
   plt.show()

   # Histogram to visualize the distribution of a numerical feature
   sns.histplot(data=df, x='numerical_feature', bins=30, kde=True)
   plt.title('Histogram of Numerical Feature')
   plt.show()
   ```


7. **🔧 Encode Categorical Variables**
    - Convert categorical variables into numeric using One-hot Encoding or Label Encoding. Ensure that the encoded values retain meaningful information.

      - **One-Hot Encoding**

        One-Hot Encoding converts categorical variables into a binary matrix. Each category value is converted into a new binary column. This method is useful when the categorical variable is nominal (i.e., there is no           ordinal relationship between the categories).

      - **When to Use**

        - The categorical variable is nominal.
        - There are no inherent relationships among the categories.

      - **Example**

        ```python
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder

        # Example data
        data = pd.DataFrame({
        'color': ['red', 'green', 'blue', 'green', 'red'],
        'size': ['S', 'M', 'L', 'M', 'S']})

        # Using OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        onehot_encoded_features = encoder.fit_transform(data[['color', 'size']])

        # Get feature names
        feature_names = encoder.get_feature_names_out(['color', 'size'])

        # Convert to DataFrame
        onehot_encoded_df = pd.DataFrame(onehot_encoded_features, columns=feature_names)

        print("One-Hot Encoded Features:")
        print(onehot_encoded_df)
        ```

      - **Label Encoding**
       
          Label Encoding converts categorical variables into numeric values by assigning a unique integer to each category. This method is useful when the categorical variable is ordinal (i.e., there is an ordinal relationship between the categories).

      - **When to Use**
          
        - The categorical variable is ordinal.
        - There is a meaningful order or ranking among the categories.
       
      - **Example**
          
         ```python
         import pandas as pd
         from sklearn.preprocessing import LabelEncoder

        # Example data
        data = pd.DataFrame({
        'color': ['red', 'green', 'blue', 'green', 'red'],
        'size': ['S', 'M', 'L', 'M', 'S']})

        # Using LabelEncoder for 'color' and 'size'
        label_encoder_color = LabelEncoder()
        label_encoded_color = label_encoder_color.fit_transform(data['color'])

        label_encoder_size = LabelEncoder()
        label_encoded_size = label_encoder_size.fit_transform(data['size'])

        # Convert to DataFrame
        label_encoded_df = pd.DataFrame({
        'color': label_encoded_color,
        'size': label_encoded_size})

        print("Label Encoded Features:")
        print(label_encoded_df)
        ```

8. **🛠️ Handle Missing Values**
    - Address missing values before Exploratory Data Analysis (EDA).
    - Visualize relationships between features using charts.
    - Fill missing values using methods like median, mean, mode, or by predicting them using models.
    ```python
    df['column'] = df['column'].fillna(df['column'].mean())
    df = df.drop('column', axis=1) or df.drop('column', inplace=True)
    df = df.dropna() or df.dropna(inplace=True)
    ```

9. **🌳 Predicting Missing Values with Random Forest**

    **Target Missing Value:**
    ```python
    df_copy = df.copy()
    target_column = 'target'
    df_not_missing = df_copy[df_copy[target_column].notna()]
    df_missing = df_copy[df_copy[target_column].isna()]

    X = df_not_missing.drop(target_column, axis=1)
    y = df_not_missing[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X_train, y_train)

    print(f'Model Accuracy: {rfr.score(X_test, y_test)}')

    X_missing = df_missing.drop(target_column, axis=1)
    predicted_values = rfr.predict(X_missing)

    df_copy.loc[df_copy[target_column].isna(), target_column] = predicted_values

    print(df_copy)
    ```

    **Feature Missing Value:**
    ```python
    df_copy = df.copy()
    features_to_impute = ['feature1', 'feature2']
    for feature in features_to_impute:
        df_not_missing = df_copy[df_copy[feature].notna()]
        df_missing = df_copy[df_copy[feature].isna()]
        
        X = df_not_missing.drop(features_to_impute, axis=1)
        y = df_not_missing[feature]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(X_train, y_train)
        
        print(f'Model Accuracy（features {feature}）: {rfr.score(X_test, y_test)}')
        
        X_missing = df_missing.drop(columns=features_to_impute)
        predicted_values = rfr.predict(X_missing)
        
        df_copy.loc[df_copy[feature].isna(), feature] = predicted_values
    
    print(df_copy)
    ```

10. **📉 Outlier Detection and Handling**
    - Detect outliers and decide whether to remove or treat them.
    - Utilize the [`pyod`](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/General%20Helpers/Machine%20Learning/Data%20Preprocessing/Handle%20outliers%20using%20pyod.py) library for outlier detection.
    - Using the [Statistic Method](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/General%20Helpers/Machine%20Learning/Data%20Preprocessing/Handle%20outliers%20using%20statistic.py), such as IQR (Interquartile Range), helps to detect and handle outliers by providing a robust measure of statistical dispersion. This method identifies outliers as values that fall below the lower bound (Q1 - 1.5 * IQR) or above the upper bound (Q3 + 1.5 * IQR), ensuring that extreme values do not disproportionately influence the dataset, thus maintaining the integrity and accuracy of the data analysis.

11. **🔨 Preprocessing and Feature Engineering**
    - Perform train-test split (typically 20-30% test size, `random_state=42`).
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42
    ```
    
12. **🗄️ Handling Imbalanced Data**
    - Use SMOTE or ADASYN for balancing.
    ```python
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    ```

13. **📏 Scaling the Data**
    - Use `StandardScaler` or `MinMaxScaler`.
    ```python
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

14. **📉 Dimensionality Reduction**
    - Apply PCA if the data has high dimensions.
    ```python
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    ```

15. **🔗 Building Pipelines**
    ```python
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    ```

16. **🧠 Model Selection and Tuning**
    - Experiment with various models (e.g., XGBoost, RandomForest).
    - Use `GridSearchCV` or Bayesian Optimization for hyperparameter tuning.

17. **🌐 Model Evaluation**
    - For regression: use MAE, MSE, RMSE.
    - For classification: use Confusion Matrix, Classification Report, ROC Curve.

18. **🧩 Clustering**
    - Common algorithms: KMeans, DBSCAN.
    - Use silhouette score to evaluate clustering performance.
    ```python
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_X)
    labels = kmeans.labels_
    ```

    ```python
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_X)
    mask = clusters != -1
    if len(np.unique(clusters[mask])) > 1:
        silhouette_avg = silhouette_score(scaled_X[mask], clusters[mask])
        print(f'The average silhouette_score is: {silhouette_avg}')
    else:
        print('Not enough clusters to calculate the meaningful silhouette_score')
    ```

## 🏁 Conclusion
 - This workflow provides a general guideline for conducting a data science project.
 
 - The specific steps may vary based on the nature of the data and the project requirements. Adjust the steps as necessary to fit your project's needs.
