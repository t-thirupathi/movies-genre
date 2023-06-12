#!/usr/bin/env python
# coding: utf-8

import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from flask_api import status
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)
FILE_PATH = Path(__file__).resolve().parent
MODELS_DIR = FILE_PATH / '../models'

# Load the models
mlb = joblib.load(MODELS_DIR / 'mlb.joblib')
encoder_model = joblib.load(MODELS_DIR / 'encoder_model.joblib')
classifier_model = joblib.load(MODELS_DIR / 'classifier_model.joblib')

@app.route('/', methods=['POST'])
def predict():
    overview = request.form.get('overview')
    if not overview:
        return 'overview missing', status.HTTP_400_BAD_REQUEST

    X = encoder_model.transform([overview])
    prediction = classifier_model.predict(X)
    prediction = mlb.inverse_transform(prediction)[0]

    return jsonify({'genre': prediction})


if __name__ == '__main__':
    app.run(port=8005)


