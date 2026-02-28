import numpy as np
from sklearn.neighbors import NearestNeighbors

def calculate_neighborhood_hit(features, labels, k=5):
    """
    Calculate the Neighborhood Hit (NH) metric.
    
    This function evaluates how well a lower-dimensional embedding preserves
    the class consistency of data points.

    Args:
        features (np.ndarray): The feature matrix (e.g., PCA or t-SNE embeddings).
        labels (np.ndarray): The corresponding class labels.
        k (int): The number of nearest neighbors to consider. Default is 5.
        
    Returns:
        float: The average Neighborhood Hit score.
    """
    if len(features) < k + 1:
        return 0.0
        
    try:
        # Find k+1 nearest neighbors (including the point itself)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
        _, indices = nbrs.kneighbors(features)
        
        # Exclude the first column (the point itself)
        neighbor_indices = indices[:, 1:]
        
        # Extract labels of neighbors
        neighbor_labels = labels[neighbor_indices]
        
        # Compare physical labels
        hits = (neighbor_labels == labels[:, None])
        
        # Average hits per point, then average over all points
        nh_score = np.mean(np.mean(hits, axis=1))
        return float(nh_score)
    except Exception as e:
        print(f"Warning: Could not calculate NH ({e}).")
        return 0.0
