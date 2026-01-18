from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class ConditionedKNeighborsRegressor(BaseEstimator, RegressorMixin):
    """
    KNN Regressor that conditions on a categorical variable (Primary Site).
    It fits a separate KNeighborsRegressor for each unique value in the last column of X.
    """
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.models = {}
        self.n_features_in_ = None
        self.n_outputs_ = None

    def fit(self, X, y):
        """
        Fit the model.
        Args:
            X: array-like of shape (n_samples, n_features + 1)
               The last column MUST be the site index (categorical).
            y: array-like of shape (n_samples, n_outputs)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Split features and site index
        X_feat = X[:, :-1]
        sites = X[:, -1].astype(int)
        
        self.n_features_in_ = X_feat.shape[1]
        self.unique_sites = np.unique(sites)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs_ = y.shape[1]
        
        for site in self.unique_sites:
            mask = (sites == site)
            X_subset = X_feat[mask]
            y_subset = y[mask]
            
            # Handle small class sizes
            k = min(self.n_neighbors, len(X_subset))
            if k < 1: 
                # Should not happen if data is clean, but safety check
                continue
                
            knn = KNeighborsRegressor(
                n_neighbors=k,
                weights=self.weights,
                metric=self.metric
            )
            knn.fit(X_subset, y_subset)
            self.models[site] = knn
            
        return self

    def predict(self, X):
        """
        Predict targets.
        Args:
            X: array-like of shape (n_samples, n_features + 1)
        """
        X = np.asarray(X)
        X_feat = X[:, :-1]
        sites = X[:, -1].astype(int)
        
        # Initialize output
        predictions = np.zeros((X.shape[0], self.n_outputs_))
        
        # Group by site for efficiency
        query_sites = np.unique(sites)
        
        for site in query_sites:
            if site not in self.models:
                # If site was seen in test but not train, we can't predict with conditioned model.
                # Fallback strategies:
                # 1. Use global mean (simple)
                # 2. Use nearest neighbor from ALL data (requires keeping a global model)
                # For this experiment, we assume strict splitting where sites match.
                # We leave zeros (or could raise warning).
                continue
            
            mask = (sites == site)
            preds = self.models[site].predict(X_feat[mask])
            predictions[mask] = preds
            
        if self.n_outputs_ == 1:
            return predictions.ravel()
        return predictions

    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": self.metric
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
