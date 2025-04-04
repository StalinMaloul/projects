from data_task01 import corpus
from time import time
from k_means import KMeansClusters
from hierarchical_cluster import HierarchicalClusters
from cluster_utils import ClusterGridSearch, create_pipeline, plot_scores

max_data = 100
file_path = "sorted_data_acl"

X = corpus(file_path, max_data)

model_names = []
pipelines = []

for model in (KMeansClusters(), HierarchicalClusters()):
    for vectorizer in ("onehot", "tfidf"):
        pipelines.append(create_pipeline(model, vectorizer_name=vectorizer))
        model_names.append(type(model).__name__ + " " + vectorizer)

parameters = [
    {
        "n_clusters": (2, 4, 8),
    },
    {
        "n_clusters": (2, 4, 8),
    },
    {
        "n_clusters": (2, 4, 8),
    },
    {
        "n_clusters": (2, 4, 8),
    },
]

scores = []

for model_name, pipeline, parameter_grid in zip(model_names, pipelines, parameters):
    grid_search = ClusterGridSearch(pipeline, parameter_grid)

    t0 = time()
    clusters = grid_search.fit_transform(X)
    print(f"Done in {time() - t0:.3f}s")

    print("Best parameters combination found: ")
    best_parameter, best_value = grid_search.get_best_parameters()
    print("n_clusters: {}".format(best_parameter))

    scores.append((model_name + " n_clusters={}".format(best_parameter), best_value))

for score in scores:
    print("Model name ", score[0])
    print("Silhouette score: {:.3f}".format(score[1]))

plot_scores(scores)
