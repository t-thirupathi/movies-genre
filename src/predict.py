import joblib
from flask import Flask, request, jsonify
from flask_api import status
from sklearn.preprocessing import MultiLabelBinarizer
from config import load_config

config = load_config()
app = Flask(__name__)

# Load the models
mlb: MultiLabelBinarizer = joblib.load(config['MODELS_DIR'] / 'mlb.joblib')
encoder_model = joblib.load(config['MODELS_DIR'] / 'encoder_model.joblib')
classifier_model = joblib.load(config['MODELS_DIR'] / 'classifier_model.joblib')

@app.route('/', methods=['POST'])
def predict() -> str:
    """
    Endpoint to predict the genre based on the provided overview text.
    Returns:
        A JSON response containing the predicted genre.
    """
    overview = request.form.get('overview')
    if not overview:
        return 'overview missing', status.HTTP_400_BAD_REQUEST

    X = encoder_model.transform([overview])
    prediction = classifier_model.predict(X)
    prediction = mlb.inverse_transform(prediction)[0]

    return jsonify({'genre': prediction})


if __name__ == '__main__':
    app.run(port=8005)


