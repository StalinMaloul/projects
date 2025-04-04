from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.cluster.util import cosine_distance

class KMeansClusters(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=7):
        self.k = n_clusters
        self.distance = cosine_distance
        self.model = KMeansClusterer(self.k, self.distance, conv_test=1e-1, avoid_empty_clusters=True)

    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        return self.model.cluster(documents, assign_clusters=True)
    
    def set_params(self, parameters):
        self.model = KMeansClusterer(parameters['n_clusters'], self.distance, conv_test=1e-1, avoid_empty_clusters=True)