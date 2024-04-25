# 🖥️ E-Commerce 💼 #

## 📘 Project Overview:
 * This is a transnational data set which contains all the transactions occurring between 2010/12/01 and 2011/12/09 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
   
 * This project focuses on analyzing e-commerce data through unsupervised machine learning models. The primary goal is to identify distinct customer segments based on their purchasing behavior, enhancing targeted marketing strategies.

## 🛠️ Skills Used:
### 🐍 Python Programming and Data Handling Skills:
* Pandas
* Numpy
* Matplotlib
* Seaborn
* StandardScaler
* Pipeline
* PyOD
* PCA
* Silhouette Score
### 🕹️ Unsupervised Machine Learning:
* KMeans 
* DBSCAN

## 📊 Skills Detail:
### 👓 Data Handling and Visualization:
* Python Libraries: Used Pandas for data manipulation and analysis, NumPy for numerical operations, and Matplotlib and Seaborn for creating diverse plots like histograms, box plots and more  to visualize data distributions and inter-variable relationships.

### 🪄 Data Preprocessing:  
 - **Implementation with the PyOD Library**: For outlier detection, the PyOD library was utilized, providing access to several powerful outlier detection algorithms. Specifically, methods like Isolation Forest (IForest), K-Nearest Neighbors (KNN), Elliptical Envelope (ECOD), and Local Outlier Factor (LOF) were employed.
   
  - **Each of these algorithms offers a unique approach to identifying outliers**:
    - **IForest**: Isolates anomalies instead of profiling normal data points, which is efficient for handling high-dimensional datasets.
      
    - **KNN**: Detects outliers by measuring the distance from a point to its neighbors, identifying those points that have a significantly longer distance.
     
    - **ECOD**: Assumes normal data points follow an elliptical distribution, effectively identifying outliers that deviate from this fit.
      
    - **LOF**: Measures local deviations of density compared to neighbors, which is effective in identifying outliers in a clustered dataset.
      
    - **PCA**: Reduces the dimensionality of the data, highlighting anomalies as those points that have large variations in the reduced dimensions.
   
 - **RFM Analysis**: Conducted Recency, Frequency, and Monetary (RFM) analysis to quantify customer value.
    - **Recency(R)**: Measures how recently a customer has made a purchase. This metric helps identify customers who have engaged with the brand recently, suggesting they are more likely to respond to 
                      new offers.

    - **Frequency(F)**: Indicates how often a customer makes a purchase within a fixed time period. Frequent buyers are often more engaged and potentially more loyal.
  
    - **Monetary Value(M)**: Reflects the total money spent by a customer over a period. Higher monetary values indicate customers who contribute more to the revenue and can be targeted for upselling                                 and cross-selling opportunities.

### 🔬 Feature Engineering and Dimensionality Reduction:
  - **PCA (Principal Component Analysis)**: Implemented to reduce the number of dimensions in the dataset, with the optimal number of components determined through visual analysis and the elbow method.

  - **Pipeline Usage**: Integrated data preprocessing steps into a pipeline to ensure consistency and efficiency in processing.

### ⚙️ Machine Learning and Cluster Analysis:
  - **Model Algorithm**:
    - **KMeans**: KMeans is a partitioning clustering method that divides data into non-overlapping subsets without internal structure. It starts with a random selection of cluster centers and         iteratively assigns points to the nearest cluster, updating the centers based on the mean of the points.

    - **DBSCAN**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters as areas of high density separated by areas of low density, without requiring the number of clusters to be predefined. It effectively classifies outliers as noise, handling arbitrarily shaped clusters.
   
  - **Model Selection and Tuning**:
    - **KMeans + PCA**: Applied KMeans clustering with Principal Component Analysis to reduce dimensionality and enhance feature focus. Adjustments were made based on silhouette scores to achieve optimal clustering, **resulting in a silhouette score of 0.48**.
   
    - **DBSCAN Clustering**: Implemented DBSCAN focusing directly on raw data by adjusting the eps and min_samples parameters, which improved clustering performance and achieved a **silhouette score of 0.62**.
   
  - **Further Analysis**:
    - **Cluster Mean Analysis**: Calculated cluster means using DBSCAN predictions, categorizing data points into clusters labeled as -1 (noise), 0, and 1.

    - **Noise Analysis**: Conducted a deeper examination of noise and non-noise clusters to understand the characteristics that differentiate these data points.
      
    - **Employed scatter plots to visually distinguish between noise and non-noise clusters, aiding in the interpretation of clustering results**.

### 📏 Model Assessment:
  - **Silhouette Score**:The Silhouette Score is a metric used to evaluate the quality of clustering in a dataset. It measures how similar an object is to its own cluster compared to other clusters. The score ranges from -1 to 1, where a high value indicates that the object is well-matched to its own cluster and poorly matched to neighboring clusters, thus signifying well-separated clusters.

## 🌐 Conclusion:
* The E-commerce Cluster Analysis Project effectively utilized RFM analysis, PCA, and unsupervised machine learning methods like KMeans and DBSCAN to segment customers.
  
* By applying silhouette scores to assess the clustering quality, the project optimized customer segmentation, enhancing marketing strategies
  
* This method demonstrated how a detailed analysis of clustering and noise can significantly enhance e-commerce strategies.







