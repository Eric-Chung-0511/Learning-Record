# 💫 Project Highlights

<a name="top"></a>

## 📋 Table of Contents
- [Data Analysis](#data-analysis)
- [Supervised Machine Learning](#supervised-machine-learning)
- [Unsupervised Machine Learning - Clustering](#unsupervised-machine-learning-clustering)
- [Artificial Neural Networks (ANNs)](#artificial-neural-networks-anns)
- [BERT and RoBERTa](#bert-and-roberta)
- [Future Goals and Directions](#future-goals-and-directions)
- [Data Science Project WorkFlow](#data-science-project-workflow)

<h2 id="data-analysis">📊 Data Analysis</h2>

* In my **Exploration of Data Analysis**, I utilized these diagrams has enabled a deeper dive into the data, revealing intricate relationships and patterns. These visual tools, such as scatter plots and heat maps, are crucial for uncovering trends, identifying outliers, and understanding data distributions. This approach enhances data interpretability and supports more informed decision making.

* Delving further into the realm of **feature engineering**, I have meticulously implemented feature selection and transformation to bolster the predictive capabilities of my models. My repositories include a broad array of feature extraction techniques such as variable transformations, interaction feature creation, and **Principal Component Analysis (PCA)** for dimensionality reduction. 

* Additionally, I have utilized a range of methods like **encoding categorical variables**, imputing missing values, and constructing derived features using domain knowledge. These strategies have greatly improved the strength and performance of algorithmic models.

* A critical component of my analytical repertoire is the detection and handling of outliers, which can skew results and impede model accuracy. To address this, I've employed the robust algorithms provided by the **PyOD** library, identifying and prudently managing these anomalies to maintain the integrity of the analysis.

* I employed both **SMOTE (Synthetic Minority Over-sampling Technique) and ADASYN (Adaptive Synthetic Sampling Approach)** to address data imbalance issues in the dataset. These methods help mitigate the problems associated with learning from imbalanced data by artificially synthesizing new examples from the minority class.
  
  - **SMOTE** works by creating synthetic samples from the minority class. It does this by finding the k-nearest neighbors for each minority class sample and interpolating new samples between the original ones and their neighbors. This helps to enhance the decision boundary’s definition without losing valuable data.
    
  - **ADASYN**, similar to SMOTE, also generates synthetic samples from the minority class but with an additional adaptive component. It places more synthetic points in regions where the classifier is more likely to make errors. This is particularly useful for enhancing the model's ability to generalize from those underrepresented data points that are close to the decision boundary.
    
  - **Both methods not only balance the class distribution effectively but also enhance the robustness and accuracy of classification models in highly skewed datasets.**

**[⇧ back to top ⇧](#top)**

<h2 id="supervised-machine-learning">⚙️ Supervised Machine Learning</h2>

* In the field of machine learning, my expertise is not limited to traditional models like **Logistic Regression**, **Decision Trees**, and **KNN** for solving classification problems.

* I've also explored more complex models, such as **Support Vector Machine (SVM)**, **Random Forest**, and **Gradient Boosting Machines (GBM)**. These are advanced methods for tackling complex nonlinear problems.
  
  - **Support Vector Machine (SVM)** is a powerful supervised learning algorithm used primarily for classification and regression challenges. SVM operates by identifying the optimal hyperplane in a high-dimensional space that best separates different classes. This is achieved through the maximization of the margin between data points of the classes, making SVM particularly effective in high dimensional spaces.
    
  - **Random Forest** is an ensemble learning technique that builds multiple decision trees and merges them together to obtain a more accurate and stable prediction. It works by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random Forest is known for its robustness against overfitting, especially in cases where the dataset is very large.
    
  - **Gradient Boosting Machines (GBM)**, These techniques build the model in stages, and generalize them by allowing optimization of an arbitrary differentiable loss function. **XGBoost**, an extension of gradient boosting, provides a scalable and efficient implementation of gradient boosting that has proven effective across a variety of data science competitions and challenges.

* For precise model tuning, techniques such as:
  - **Cross-Validation**, which systematically divides the dataset into subsets to ensure the model's performance is validated thoroughly, enhancing its generalizability.
    
  - **GridSearchCV**, which tests a variety of parameter combinations, streamlining the process to determine the most effective configurations for a model. This approach automates the tuning process, ensuring that the optimal parameters are selected based on systematic evaluation and comparison.
    
  - **Bayesian Optimization**, which uses probabilistic models to optimize parameter selection more efficiently than traditional methods like GridSearchCV by intelligently predicting and focusing on the most promising parameter combinations. This approach significantly reduces the number of evaluations needed, prioritizing those that are likely to yield the best performance, which is especially beneficial when evaluations are computationally expensive.

* Using models like **ARIMA**, **SARIMA**, and **SARIMAX** for predicting time series data is crucial for understanding and forecasting trends in areas like the stock market and sales volumes. **SARIMA** and **SARIMAX** account for various seasonal patterns and external influences, making them highly effective for data with trends and seasonal variations.

* I have initiated a project focused on **Natural Language Processing (NLP)**, specifically aimed at predicting Myers-Briggs Type Indicator (MBTI) personality types. This project leverages textual analysis to identify personality traits based on users' written expressions.

* Finally, by precisely evaluating model performance using tools like confusion matrices, **ROC curves**, and **Precision-Recall Curves**, I can thoroughly understand the strengths and weaknesses of models, thereby making wiser decisions in real world problems.

**[⇧ back to top ⇧](#top)**

<h2 id="unsupervised-machine-learning-clustering">🎛️ Unsupervised Machine Learning - Clustering</h2>

* In the realm of unsupervised learning, I have already begun implementing techniques like **KMeans** and **DBSCAN**, which play crucial roles in discovering inherent structures within unlabelled data.
  
  - **KMeans** excels in identifying clusters based on centroid proximity, making it particularly suitable for applications that require partitioning of data into distinct groups, such as customer segmentation.
    
  - **DBSCAN** is a density-based clustering algorithm, suitable for identifying groups of arbitrary shapes and capable of recognizing noise points.

* Looking ahead, future advancements will focus on scaling these algorithms to handle larger datasets more efficiently and enhancing their ability to discern more subtle patterns and relationships in the data. This ongoing development and deeper integration of clustering techniques into my analytical toolkit will continue to enhance the robustness and scope of my data analysis capabilities.

**[⇧ back to top ⇧](#top)**

<h2 id="artificial-neural-networks-anns">🤖 Artificial Neural Networks (ANNs)</h2>

* ANNs are powerful computational models inspired by the human brain's structure, making them exceptionally adept at recognizing subtle patterns and anomalies in large datasets.
  
  - **Application in Fraud Detection**: ANNs excel in detecting fraudulent transactions by learning to differentiate between legitimate and fraudulent behaviors through training on a dataset of transaction records. This capability is critical in financial security, where accurate detection can prevent substantial financial losses.
    
  - **Architecture and Function**: In this project, the ANN is structured with multiple layers, including an input layer, several hidden layers, and an output layer that employs a sigmoid activation function. This setup is tailored to effectively handle the binary classification task of distinguishing fraudulent transactions.
    
  - **Training and Optimization**: The network is trained using a backpropagation algorithm with a binary cross-entropy loss function, which refines the model by minimizing prediction errors. This process is crucial for enhancing the ANN’s ability to accurately identify fraudulent activities in transaction data.
 
**[⇧ back to top ⇧](#top)**

<h2 id="bert-and-roberta">📄 BERT and RoBERTa</h2>

### Overview
* BERT is a powerful pre-trained language model developed by Google, designed to understand the context of a word in search queries. Unlike traditional models that read text input sequentially, BERT reads text bidirectionally, capturing the context from both the left and the right sides of a word. This capability makes BERT exceptionally effective for various NLP tasks, including text classification, question answering, and named entity recognition.

### RoBERTa
* RoBERTa, developed by Facebook, builds upon BERT by making several optimizations. These include training with larger batches, using longer sequences, and training on more data. These enhancements make RoBERTa more robust and capable of achieving even higher performance on NLP benchmarks compared to BERT.

### Application in SkimLit
- **Text Classification and Summarization**: In the SkimLit project, the RoBERTa model was fine-tuned to automatically classify and summarize biomedical literature. This helps researchers quickly access relevant information from vast amounts of scientific papers.
- **Architecture and Function**: The RoBERTa model processes tokenized input text and outputs high-dimensional vectors representing the semantic meaning of the text. This model is structured with multiple layers, including an embedding layer and several transformer layers that capture complex linguistic features.
- **Training and Optimization**: The model was fine-tuned using a biomedical literature dataset, employing a binary cross-entropy loss function and backpropagation for optimization. This fine-tuning process enhances the model's ability to accurately classify and summarize the abstracts.

**[⇧ back to top ⇧](#top)**


<h2 id="future-goals-and-directions">💎 Future Goals and Directions</h2>

### 🚀 Transfer Learning
* Recently, I have been exploring Transfer Learning and have begun to grasp its significant potential. **This method enables me to utilize pretrained models and fine-tune them for specific tasks, greatly reducing the time and resources required.**

* While I'm still relatively new to this technique, I'm steadily building my understanding. By tweaking the parameters of these pretrained models, I'm learning how to develop tailored models that suit my needs, making the most of the robust features and patterns learned by the original models. This approach has already demonstrated its power and efficiency.

* **Looking ahead, I am also planning to apply large language models like GPT-4 and Meta Llama3 to real-world applications. By fine-tuning these models on specific datasets and tasks, my goal is to create highly customized and effective language models that can be leveraged in practical scenarios, achieving meaningful results with greater efficiency.**
  
### 🗣🔄🤖 Natural Language Processing (NLP) and Transformer Technology
* My future goals involve leveraging advanced Transformer models to expand the scope of NLP applications. This includes tasks such as sentiment analysis, the development of intelligent recommendation systems, and the creation of interactive chatbots.

* **Additionally, I aspire to develop AI systems capable of automatically extracting and summarizing key points from academic papers, allowing researchers to spend less time on initial screening and more time on in-depth reading and exploration of relevant studies.**

* By harnessing the full potential of these cutting-edge technologies, I aim to build sophisticated AI systems capable of understanding and generating human-like text, enhancing user interaction, and delivering personalized experiences across various applications like text generation, translation, and more.

**[⇧ back to top ⇧](#top)**

<h2 id="data-science-project-workflow">🌐 Data Science Project Workflow</h2>

* Based on my existing hands-on experience, I have developed a structured [**Data Science Project Workflow**](https://github.com/Eric-Chung-0511/Learning-Record/blob/main/Data%20Science%20Projects/Data%20Science%20Project%20WorkFlow.md) that ensures my data analysis processes are thorough, systematic, and replicable. **This workflow encompasses data collection, preprocessing, exploration, modeling, and evaluation**, each step being crucial in transforming raw data into actionable insights and making data driven decisions, thereby enhancing the accuracy and effectiveness of the models I develop.
  
**[⇧ back to top ⇧](#top)**

