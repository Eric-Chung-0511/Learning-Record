# 🏪 Walmart Sales Forecasting 🏪 

## 🔍 Project Overview:
* Walmart is a renowned retail corporation that operates a chain of hypermarkets. Here, Walmart has provided a data combining of 45 stores including store information and monthly sales.
  
* The data is provided on weekly basis. Walmart tries to find the impact of holidays on the sales of store. For which it has included four holidays’ weeks into the dataset which are Christmas, Thanksgiving, Super bowl, Labor Day. Here we are owing to Analyze the dataset given.

* Main Objective is to predict sales of store in a week

## 🛠️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* calmap
### 🤖 Machine Learnig (Stats Models):
* PACF
* ACF
* Auto_ARIMA
* SARIMAX
* Adfuller
* Seasonal Decompose
* Mean Abosulte Error

## 📊 Skills Detail:
### 👓 Data Handling and Visualization:
- **Python Libraries**: Utilized Pandas, NumPy, Matplotlib, and Seaborn for data manipulation and visualization.
  
- **Exploratory Data Analysis (EDA)**: Conducted using heatmaps and calmap to visualize and explore correlations and patterns in the data over time.

### 🔄 Data Preprocessing:
- **Resampling**: Data was resampled weekly to simplify the time series and focus analysis on broader trends rather than daily fluctuations.

### 🔢 Statistical Analysis and Model Preparation:
- **Autocorrelation and Partial Autocorrelation (ACF and PACF)**: Analyzed the data using ACF and PACF plots to identify the appropriate parameters for ARIMA modeling, specifically to determine the lags of autoregression (p) and moving average (q).

- **ARIMA and SARIMAX**:
  - **ARIMA (AutoRegressive Integrated Moving Average)**: A modeling technique that forecasts a time series using its own past values, differences, and forecast errors.
    
  - **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)**: An extension of ARIMA that supports external variables and seasonal components, ideal for incorporating known influences like holidays into the model.

### 🤖 Machine Learning:
- **Model Optimization**: Utilized auto_arima from the statsmodels library to automatically discover the optimal order for the SARIMAX model, considering seasonality and external factors.

###  📏 Model Assessment:
- **Performance Evaluation**: The model's accuracy was evaluated using the Mean Absolute Error (MAE) metric, which measures the average magnitude of errors in a set of predictions, without considering their direction.

## 🎯 Conclusion:
* This project demonstrated the effectiveness of combining statistical analysis with machine learning techniques to forecast time series data.
  
* By integrating external variables and seasonality into the SARIMAX model, we were able to improve prediction accuracy significantly.
  
* The use of ACF and PACF for parameter estimation, along with auto_arima for model optimization, provided a robust framework for handling complex time series datasets.

* Despite the low MAE suggesting accurate predictions, visual assessments of the plots revealed discrepancies, highlighting the need for both numerical and visual evaluations to fully gauge model performance.

## 📚 Acknowledgments and References:
* I am grateful to the developers and researchers whose work has significantly influenced this project. Below are the resources that have been instrumental:

* https://www.kaggle.com/code/fatmayousufmohamed/time-serises-analysis#Thanks!
