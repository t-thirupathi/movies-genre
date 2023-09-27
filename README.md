End to End ML solution and deployment example

1. Machine learning model to predict genre of movies based on the text overview
2. Deployment using Docker

Steps to run:
1. pip install -r requirements.txt
2. python src/train.py
3. python src/predict.py
4. curl -d "overview=A movie about penguins in Antarctica building a spaceship to go to Mars." -X POST http://localhost:8005
