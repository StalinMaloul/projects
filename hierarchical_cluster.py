from sklearn.base import BaseEstimator, TransformerMixin
from nltk.cluster.util import cosine_distance
from sklearn.cluster import AgglomerativeClustering

class HierarchicalClusters(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=2):
        self.model = AgglomerativeClustering(n_clusters=n_clusters)

    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        clusters = self.model.fit_predict(documents)
        self.labels = self.model.labels_
        self.children = self.model.children_
        return clusters
    
    def set_params(self, parameters):
        self.model.set_params(**parameters)