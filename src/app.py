#!/usr/bin/env python
# coding: utf-8

import os, joblib
from flask import Flask, request, jsonify
from flask_api import status
from sklearn.preprocessing import MultiLabelBinarizer

app = Flask(__name__)
FILE_PATH = os.path.dirname(__file__)

# Load the models
mlb = joblib.load(FILE_PATH + '/../models/mlb.joblib')
encoder_model = joblib.load(FILE_PATH + '/../models/encoder_model.joblib')
classifier_model = joblib.load(FILE_PATH + '/../models/classifier_model.joblib')

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


