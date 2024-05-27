import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def find_optimal_k(data, random_state=42, max_k=None):
    """
    Use the Elbow Method to find the optimal number of clusters for KMeans.

    Parameters:
    data: array-like, shape (n_samples, n_features)
        The data to be clustered.
    random_state: int, optional (default=42)
        The random state for initializing the KMeans clusters.
    max_k: int, optional (default=None)
        The maximum number of clusters to consider. If None, it will use the range up to len(data).

    Returns:
    None
    """
    if max_k is None:
        max_k = len(data)  # Default to the number of samples if max_k is not provided
    
    sse = []  # List to store the sum of squared errors for each k

    # Iterate over the range of cluster numbers
    for k in range(1, max_k + 1):
        # Initialize and fit KMeans with k clusters
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)  # Append the SSE (inertia) to the list
    
    # Plot the SSE against the number of clusters
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample data
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Find the optimal number of clusters
    find_optimal_k(data, max_k=10)
