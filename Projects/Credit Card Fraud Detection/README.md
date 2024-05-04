# 💳 Credit Card Fraud Detection 🔍

## ✨ Project Overview:
* It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

* This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

## ⚙️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* StandardScaler
* IQR (Handle Outliers)
* Adasyn
* classification Report
* AUC_ROC Score
### 🧠 Deep Learning:
* ANN (Artificial Neural Network)

## 🤖 Skills Detail:
### 👓 Data Handling and Visualization:
* Python Libraries: Used Pandas for data manipulation and analysis, NumPy for numerical operations, and Matplotlib and Seaborn for creating diverse plots like histograms, box plots and more  to visualize data distributions and inter-variable relationships. These visualizations help highlight key features and relationships critical to understanding fraud dynamics.

### ⚒️ Data Preprocessing:
* **Log Transformation on 'Amount'**: To address skewed distributions typically seen with transaction amounts in fraud data, applied a logarithmic transformation to the 'Amount' feature. This helps normalize the data, reducing the influence of extreme values on the analysis.
  
* **Innovative Outlier Handling with IQR**: The Interquartile Range (IQR) is a statistical measure used to identify the spread of the middle 50% of data points in a dataset. It is calculated as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data. The IQR helps in identifying outliers by defining thresholds. Typically, any data point that lies more than 1.5 times the IQR below Q1 or above Q3 is considered an outlier.
  
* **Instead of simply removing outliers, implemented a method to handle them by replacing with the median or capping based on quantiles.** This approach, crucial in imbalanced datasets like fraud detection, helps maintain essential data points and avoids bias towards the majority class.

* **Data Splitting**: Segregated the data into training, validation, and test datasets to ensure a thorough evaluation phase.

* **Feature Scaling with StandardScaler**: Utilized StandardScaler to normalize the features within the dataset, ensuring that each feature contributes equally to the analysis and subsequent model predictions. This is crucial in algorithms that are sensitive to the scale of input data, like most machine learning algorithms, because it enhances their performance by providing a level playing field.

* **Balancing data with ADASYN**:
  - **Principle of ADSYN**: ADASYN (Adaptive Synthetic Sampling) is a technique used to create synthetic samples for the minority class in an imbalanced dataset. The fundamental idea behind ADASYN is to use a weighted distribution for different minority class samples according to their level of difficulty in learning, where more synthetic data is generated for harder samples. Specifically, it focuses on generating new samples next to the existing minority samples that are wrongly classified using a k-nearest neighbors algorithm.

  - **Differences Between ADASYN and SMOTE**:
    - **Sample Generation Focus**: While both ADASYN and SMOTE generate synthetic samples by interpolating between existing minority class samples, ADASYN places more emphasis on those samples that are difficult to classify. SMOTE generates the same number of samples for each minority class instance, treating all instances equally.
      
    - **Adaptive Nature**: ADASYN adapts to the inherent data structure. It uses a density distribution to automatically decide the number of synthetic samples needed for each minority sample, particularly focusing on the regions where the classifier has performed poorly, such as where the samples are misclassified.
    
    - **Handling of Borderline Examples**: ADASYN tends to focus more on the borderline examples, making it better suited for dealing with problems where the classes are not only imbalanced but also have complex decision boundaries.
  - **Why Use ADASYN**:
    - **Targeting Difficult Cases**: ADASYN is particularly effective in generating synthetic samples for those cases that are difficult to learn, which is crucial in complex scenarios like fraud detection where fraudulent behaviors are diverse and hard to predict.
      
    - **Boosting Minority Class Performance**: By focusing on the harder-to-classify, near-boundary samples, ADASYN improves classification accuracy for the minority class, which is essential in areas such as fraud detection where detecting rare events accurately is critical.
      
    - **Adapting to Complex Data Structures**: ADASYN excels in scenarios with complex decision boundaries and high class imbalance, thanks to its ability to adaptively generate more samples where they are most needed, enhancing the model's overall robustness and effectiveness.
   
### 🧬 Artificial Neural Network(ANN):
* **Structure**: An Artificial Neural Network (ANN) is a computational model inspired by the structure and functions of biological neural networks. It consists of layers of nodes, each node representing a neuron, and connections between these nodes representing synapses.

* **Information Flow**: Information flows through the network from input to output. Each node processes the input and passes it on to the next layer until the output layer is reached.
  
* **Activation Functions**: In the output layer, ANNs typically use a sigmoid activation function to map predictions to a probability distribution. This is particularly ideal for binary classification tasks such as fraud detection, where the model predicts either a 'fraud' or 'not fraud' outcome.
  
* **Loss Functions and Optimizers**: The choice of loss function and optimizer is crucial for training an ANN effectively. A binary cross-entropy loss function is commonly chosen in conjunction with the sigmoid activation. This loss function calculates the error rate between the predicted probabilities and the actual class outputs.
  
* **Backpropagation**: The calculated error is then used during backpropagation, a key mechanism in ANN training. This process involves adjusting the weights of the connections in the network in order to minimize prediction errors, enhancing the model's accuracy over iterations, and the goal is find the optimal function.

### 🧭 Model Evaluation:
* **Precision-Recall Curve Analysis**: Plotted precision-recall curves to find the optimal decision making threshold, which balances recall (sensitivity) and precision, critical in fraud detection where missing a fraudulent transaction (low recall) is more detrimental than a false positive.
  
* **Phased Testing Approach**: Validated the model on the validation set followed by the test set, using these separate datasets to gauge the model’s ability to generalize to new, unseen data.
  
* **Comprehensive Performance Metrics**: Utilized classification reports to provide detailed insights into the model's accuracy, precision, recall, and F1-score on both validation and test data, giving a rounded view of model effectiveness.

## 🎯 Conclusion:
* The project elucidated the importance of preprocessing in handling imbalanced datasets and the effectiveness of ANN in detecting fraudulent transactions. The comparative analysis between different outlier handling methods provided insights into how data preparation significantly affects the outcome, underscoring the need for tailored approaches in fraud detection.

* Particularly in the context of credit card fraud, achieving a high recall is critical, as it is essential to detect as many fraudulent transactions as possible. Missing a fraudulent transaction can have significant financial implications, thus emphasizing recall ensures that fewer fraud cases go undetected, prioritizing the security and trust of the financial system.

## 📚 Acknowledgments and References:
* I am grateful to the developers and researchers whose work has significantly influenced this project. Below are the resources that have been instrumental:

* https://www.kaggle.com/code/idalez/creditcardfraud-analysis-and-prediction-using-ann
