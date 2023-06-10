#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os, ast, joblib
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import xgboost as xgb

FILE_PATH = os.path.dirname(__file__)

def preprocess_data():
    df = pd.read_csv(FILE_PATH + '/../data/movies_metadata.csv', low_memory=False)
    df = df[['overview', 'genres']]
    df.dropna(inplace=True)
    
    # Clean up the genres column
    df['genres'] = df['genres'].apply(ast.literal_eval) \
                              .apply(lambda x: [i['name'] for i in x])
    
    # Filter out rare genres
    counter = Counter([genre for sublist in df['genres'] for genre in sublist])
    GENRE_FREQ_THRESHOLD = 10
    popular_genres = [genre for genre, count in counter.items() if count >= GENRE_FREQ_THRESHOLD]
    
    df['genres'] = df['genres'].apply(lambda genres: [genre for genre in genres if genre in popular_genres])
    
    # Drop rows with no popular genre
    df['genres'] = df['genres'].apply(lambda genres: genres if len(genres) > 0 else np.nan)
    df.dropna(subset=['genres'], inplace=True)
    
    # One hot encoding of genres
    mlb = MultiLabelBinarizer()
    df_genres = pd.DataFrame(mlb.fit_transform(df.pop('genres')),
                              columns=mlb.classes_,
                              index=df.index)
    
    joblib.dump(mlb, FILE_PATH + '/../models/mlb.joblib')
    
    df = df.join(df_genres)
    df.to_csv(FILE_PATH + '/../data/processed_movies_metadata.csv', index=False)


def train_model(X_train, y_train, encoder, multilabeler, estimator):
    if encoder == 'TfIdf':
        encoder_model = TfidfVectorizer().fit(X_train)
    elif encoder == 'SentenceTransformer':
        encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
        encoder_model.transform = encoder_model.encode
    
    X_encoded = encoder_model.transform(X_train)
    classifier = multilabeler(estimator=estimator)
    classifier.fit(X_encoded, y_train)
    return (encoder_model, classifier)

if __name__ == '__main__':
    print('Preprocessing data...')
    preprocess_data()

    df = pd.read_csv(FILE_PATH + '/../data/processed_movies_metadata.csv')
    X = df['overview']
    y = df.drop(['overview'], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Reset indices so that SentenceTransformer encode doesn't complain
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    print('Training model...')
    encoder = 'TfIdf'
    multilabeler = OneVsRestClassifier
    estimator = DecisionTreeClassifier()
    
    encoder_model, classifier_model = train_model(X_train, y_train,
                                           encoder, multilabeler, estimator)
    y_pred = classifier_model.predict(encoder_model.transform(X_test))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("F1 Score: %.4f"%(report['micro avg']['f1-score']))
    print("Hamming Loss: %.4f"%(hamming_loss(y_test, y_pred)))
    
    print('Saving model...')
    joblib.dump(encoder_model, FILE_PATH + '/../models/encoder_model.joblib')
    joblib.dump(classifier_model, FILE_PATH + '/../models/classifier_model.joblib')
    

