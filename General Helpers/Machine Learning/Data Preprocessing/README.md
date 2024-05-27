#  📚 Function Overview

### Handle outliers using pyod([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Loan%20Prediction/Loan%20Prediction%20_Eric.ipynb))
- **This function consists of three functions aimed at handling outliers in your dataset.**
- **detect_outliers**: Handle outliers using pyod.
- **find_best_threshold**: Find the optimal contamination threshold for outlier detection models using cross-validation based on F1 scores.
- **combined_outliers**: Combine outlier detection results from multiple models using strategies such as 'union' or 'intersection' for robust analysis.

### Handle outliers using statistics([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Human%20Activity%20Recognition%20with%20Smartphones/Human%20Activity%20Recognition%20with%20Smartphones%20_Eric.ipynb))
- **This function consists of five functions aimed at detecting and handling outliers in your dataset.**
- **handle_outliers**: Detect and handle outliers using 'median' and 'quantile' methods.
- **outlier_transformer_median**: Create a transformer to handle outliers using the median method.
- **outlier_transformer_quantile**: Create a transformer to handle outliers using the quantile method.
- **create_outlier_transformer**: Generate a function transformer based on the specified method for handling outliers.
- **create_pipeline**: Create a machine learning pipeline with outlier handling and scaling.

