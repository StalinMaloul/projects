import os
from bs4 import BeautifulSoup


def corpus(file_path, max_data):
    all_reviews = []
    files = ["positive.review", "negative.review"]
    for root, files in os.walk(file_path):
        for file in files:
            file_path_2 = os.path.join(root, file)
            if not os.path.exists(file_path_2):
                continue
            with open(file_path_2, "r", encoding="utf-8") as f:
                reading = f.read()
                soup = BeautifulSoup(reading, "html.parser")
                reviews = soup.find_all("review_text")
                for review in reviews:
                    all_reviews.append(review.text)
    return all_reviews[:max_data]