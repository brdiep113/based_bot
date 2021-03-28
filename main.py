import numpy as np
import pandas as pd
from process_tweets import *
from matplotlib import pyplot as plt
import nltk
from collections import defaultdict
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time


def process_text(df):
    """

    :rtype: object
    """
    features = df['text']
    labels = df['party']
    processed_features = []
    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        # Converting to Lowercase
        processed_feature = processed_feature.lower()
        processed_features.append(processed_feature)
        print(processed_feature)
    return processed_features, labels


def evaluate_metrics(test_set, predictions):
    print(confusion_matrix(test_set, predictions))
    print(classification_report(test_set, predictions))
    print(accuracy_score(test_set, predictions))


def CorpusIterator(corpus):
    for tweet in corpus:
        yield tweet


if __name__ == '__main__':
    df_tweets = csv_to_df('tweet_data/tweets_users_with_party.csv')
    processed_features, labels = process_text(df_tweets)
    corpus = CorpusIterator(processed_features)
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(corpus)
    processed_features = processed_features.toarray()

    X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
    text_classifier = RandomForestClassifier(n_estimators=100, random_state=0, verbose=2)

    text_classifier.fit(X_train, y_train)

    predictions = text_classifier.predict(X_test)
    evaluate_metrics(y_test, predictions)
