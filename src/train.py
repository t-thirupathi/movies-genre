import pandas as pd
import numpy as np
import ast
import joblib
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
from config import load_config

def preprocess_data(data_file: str) -> tuple:
    """
    Preprocesses the data file and returns the preprocessed data.
    Args:
        data_file: The path to the data file.

    Returns:
        A tuple containing the preprocessed data: X_train, X_test, y_train, y_test.
    """
    df = pd.read_csv(data_file, usecols=['overview', 'genres'], low_memory=False)
    # Comment below line to train on all data. Takes about 15 mins
    df = df.head(1000)
    df = df.dropna()

    # Clean up the genres column
    df['genres'] = df['genres'].apply(ast.literal_eval) \
                              .apply(lambda x: [i['name'] for i in x])

    # Filter out rare genres
    counter = Counter([genre for sublist in df['genres'] for genre in sublist])
    popular_genres = [genre for genre, count in counter.items() if count >= config['GENRE_FREQ_THRESHOLD']]
    
    df['genres'] = df['genres'].apply(lambda genres: [g for g in genres if g in popular_genres])
    
    # Drop rows with no popular genre
    df['genres'] = df['genres'].apply(lambda genres: genres if len(genres) > 0 else np.nan)
    df = df.dropna(subset=['genres'])
    
    # One hot encoding of genres
    mlb = MultiLabelBinarizer()
    df_genres = pd.DataFrame(mlb.fit_transform(df.pop('genres')),
                              columns=mlb.classes_,
                              index=df.index)
    
    joblib.dump(mlb, config['MODELS_DIR'] / 'mlb.joblib')
    
    df = df.join(df_genres)
    df.to_csv(config['DATA_DIR'] / 'processed_movies_metadata.csv', index=False)

    X = df['overview']
    y = df.drop(['overview'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Reset indices so that SentenceTransformer encode doesn't complain
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.Series, y_train: pd.DataFrame, encoder: str, multilabeler: type, estimator: type) -> tuple:
    """
    Trains a model using the provided data and parameters.
    Args:
        X_train: The training data.
        y_train: The target labels.
        encoder: The encoder type.
        multilabeler: The multilabel classifier type.
        estimator: The estimator type.

    Returns:
        A tuple containing the trained encoder and classifier models.
    """
    if encoder =='TfIdf':
        encoder_model = TfidfVectorizer().fit(X_train)
    elif encoder == 'SentenceTransformer':
        encoder_model = SentenceTransformer('all-MiniLM-L6-v2')
        encoder_model.transform = encoder_model.encode
    
    X_encoded = encoder_model.transform(X_train)
    classifier_model = multilabeler(estimator=estimator)
    classifier_model.fit(X_encoded, y_train)
    return (encoder_model, classifier_model)


def main():
    global config
    config = load_config()

    print('Preprocessing data...')
    X_train, X_test, y_train, y_test = preprocess_data(config['DATA_DIR'] / 'movies_metadata.csv')

    
    print('Training model...')
    encoder = 'SentenceTransformer'
    multilabeler = OneVsRestClassifier
    estimator = DecisionTreeClassifier()
    
    encoder_model, classifier_model = train_model(X_train, y_train,
                                           encoder, multilabeler, estimator)
    y_pred = classifier_model.predict(encoder_model.transform(X_test))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("F1 Score: %.4f"%(report['micro avg']['f1-score']))
    print("Hamming Loss: %.4f"%(hamming_loss(y_test, y_pred)))

    print('Saving model...')
    joblib.dump(encoder_model, config['MODELS_DIR'] / 'encoder_model.joblib')
    joblib.dump(classifier_model, config['MODELS_DIR'] / 'classifier_model.joblib')


if __name__ == '__main__':
    main()
