# Q12: Function to predict using trained model
import joblib
import numpy as np

model = joblib.load('linear_model.pkl')

def predict_price(features):
    return model.predict([features])[0]

sample = [0.03, 0.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]
print("Predicted Value:", predict_price(sample))
