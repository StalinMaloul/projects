import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from text_normalization import TextNormalizer
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def create_pipeline(estimator, vectorizer_name):

    steps = [
        ('normalize', TextNormalizer())
    ]

    if vectorizer_name == "onehot":
        steps.append(('vectorize', CountVectorizer(tokenizer=lambda x: x, preprocessor=None, lowercase=False, token_pattern=None)))
    else:
        steps.append(('vectorize', TfidfVectorizer(tokenizer=lambda x: x, preprocessor=None, lowercase=False, token_pattern=None)))

    steps.append(('classifier', estimator))
    
    return Pipeline(steps)

def plot_scores(scores):

    models_names = [score[0] for score in scores]

    dictionary_scores = {
        'train': [score[1] for score in scores],
        'test': [score[2] for score in scores],
    }

    fig, ax = plt.subplots()

    x = np.arange(len(models_names))

    width = 0.25  # the width of the bars
    multiplier = 0

    for label, score in dictionary_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=label)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_xticks(x + width, models_names)
    ax.set_ylabel('accuracy')
    ax.set_title('classification results')
    ax.legend(ncols=3)

    plt.show()
