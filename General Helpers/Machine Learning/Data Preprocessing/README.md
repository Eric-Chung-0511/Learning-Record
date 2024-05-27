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

### find_most_common_word([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/MBTI/MBTI%20Prediction_Eric.ipynb))
- **This function consists of two functions aimed at analyzing text data to find the most common words.**
- **find_most_common_words**: Find the most common words in a specified column of a DataFrame.
- **flatten_list**: Flatten a nested list of words into a single list to facilitate word frequency analysis.

### preprocess_text([View Example](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/MBTI/MBTI%20Prediction_Eric.ipynb))
- **This function consists of one primary function aimed at preprocessing text data in a DataFrame column.**
- **preprocess_text**: Preprocess text data by removing links, punctuation, non-words, very short or long words, and optionally special words, while also handling end-of-sentence characters and converting text to lowercase.

