import os
from bs4 import BeautifulSoup


def corpus(file_path, max_data):
    all_reviews = []
    folders = ["books", "dvd", "electronics", "kitchen_&_housewares"]
    files = ["positive.review", "negative.review", "unlabeled.review"]
    for folder in folders:
        for file in files:
            file_path_1 = os.path.join(file_path, folder, file)
            if not os.path.exists(file_path_1):
                continue
            with open(file_path_1, "r", encoding="utf-8") as f:
                reading = f.read()
                soup = BeautifulSoup(reading, "html.parser")
                reviews = soup.find_all("review_text")
                for review in reviews:
                    all_reviews.append(review.text)
    return all_reviews[:max_data]
