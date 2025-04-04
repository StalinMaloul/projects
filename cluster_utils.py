import numpy as np
from sklearn.pipeline import Pipeline
from text_normalization import TextNormalizer
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from vectorizer import TFIDFVectorizer, OneHotVectorizer
import warnings

warnings.filterwarnings("ignore")


def compute_silhoutte_score(clusters, data, vectorizer):

    X_vectorized = vectorizer.fit_transform(data)

    return silhouette_score(X_vectorized, clusters, metric="cosine")


def create_pipeline(estimator, vectorizer_name):

    steps = [("normalize", TextNormalizer())]

    if vectorizer_name == "onehot":
        steps.append(("vectorize", OneHotVectorizer()))
    else:
        steps.append(("vectorize", TFIDFVectorizer()))

    steps.append(("estimator", estimator))

    return Pipeline(steps)


class ClusterGridSearch:
    def __init__(self, pipeline, parameters):
        self.parameters = parameters["n_clusters"]
        self.pipeline = pipeline
        self.best_value = -1
        self.best_parameter = -1

    def fit_transform(self, X):

        best_clusters = None

        for parameter in self.parameters:
            print("fitting with {}".format(parameter))
            self.pipeline.named_steps["estimator"].set_params({"n_clusters": parameter})
            clusters = self.pipeline.fit_transform(X)

            silhouette_avg = compute_silhoutte_score(
                clusters, X, self.pipeline.named_steps["vectorize"]
            )

            if silhouette_avg > self.best_value:
                self.best_value = silhouette_avg
                self.best_parameter = parameter
                best_clusters = clusters

        print("best parameter found:", self.best_parameter)
        return best_clusters

    def get_best_parameters(self):

        return self.best_parameter, self.best_value


def plot_scores(scores):

    models_names = [score[0] for score in scores]

    fig, ax = plt.subplots()

    x = np.arange(len(models_names))

    width = 0.25  # the width of the bars

    rects = ax.bar(x, [score[1] for score in scores], width, label="silhoutte score")

    ax.set_xticks(x, models_names)
    ax.set_ylabel("silhoutte")
    ax.set_title("clustering results")
    ax.legend()

    plt.show()
