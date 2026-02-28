import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.clustering_evaluation.metrics_utils import calculate_neighborhood_hit
import pickle

# Just a quick check to see if scaling creates different metrics for Mean and KNN
