import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

# Detect and Handle outliers
# Set two methods, median and quantile
def handle_outliers(df, method=None):
    """
    Detect and handle outliers in a DataFrame using the specified method.

    Parameters:
    df: pandas.DataFrame
        The input DataFrame containing numeric data.
    method: str, optional (default=None)
        The method to handle outliers. Options are 'median' and 'quantile'.

    Returns:
    df_copy: pandas.DataFrame
        The DataFrame with outliers handled.
    """
    df_copy = df.copy()  # Create a copy of the DataFrame to avoid modifying the original data
    
    if method is None:
        print('No method provided. Return Original Data.')
        return df_copy  # Return original DataFrame if no method is provided
    
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns  # Select numeric columns
    for col in numeric_cols:
        if method == 'median':
            # Calculate median
            median = df_copy[col].median()
            
            # Calculate Interquartile Range (IQR)
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
            upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
            
            # Replace outliers with the median value
            df_copy[col] = df_copy[col].where(~((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)),
                                              other=median)
            
        elif method == 'quantile':
            # Calculate lower and upper quantiles
            lower_quantile = df_copy[col].quantile(0.01)
            upper_quantile = df_copy[col].quantile(0.99)
            
            # Clip values to the quantile range
            df_copy[col] = df_copy[col].clip(lower_quantile, upper_quantile)
            
        else:
            print(f'Invalid method provided for {col}')  # Handle invalid method
        
    return df_copy  # Return the DataFrame with outliers handled

# Create handle outliers function transformer, so it can be used in a pipeline
def outlier_transformer_median(df):
    """
    Function transformer to handle outliers using the median method.
    
    Parameters:
    df: pandas.DataFrame
        The input DataFrame.
    
    Returns:
    pandas.DataFrame
        The DataFrame with outliers handled using the median method.
    """
    return handle_outliers(df, method='median')

def outlier_transformer_quantile(df):
    """
    Function transformer to handle outliers using the quantile method.
    
    Parameters:
    df: pandas.DataFrame
        The input DataFrame.
    
    Returns:
    pandas.DataFrame
        The DataFrame with outliers handled using the quantile method.
    """
    return handle_outliers(df, method='quantile')

def create_outlier_transformer(method):
    """
    Create a function transformer for handling outliers using the specified method.
    
    Parameters:
    method: str
        The method to handle outliers. Options are 'median' and 'quantile'.
    
    Returns:
    sklearn.preprocessing.FunctionTransformer
        The function transformer for handling outliers.
    
    Raises:
    ValueError: If an invalid method is provided.
    """
    if method == 'median':
        return FunctionTransformer(outlier_transformer_median)
    elif method == 'quantile':
        return FunctionTransformer(outlier_transformer_quantile)
    else:
        raise ValueError("Invalid method provided")

# Make sure you already define X, y and split into X_train, X_test, y_train, y_test
def create_pipeline(outlier_method, X_train):
    """
    Create a machine learning pipeline with outlier handling and scaling.

    Parameters:
    outlier_method (str): Method to handle outliers ('median' or 'quantile').
    X_train (pd.DataFrame): Training data.

    Returns:
    Pipeline: A machine learning pipeline.
    """
    outlier_transformer = create_outlier_transformer(outlier_method)
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns  # Numeric columns
    preprocessor = ColumnTransformer(transformers=[
        ('outliers', outlier_transformer, numeric_cols),
        ('scaler', StandardScaler(), numeric_cols)
    ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95)),
        ('classifier', 'passthrough')
    ])
    
    return pipeline
