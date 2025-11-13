# Q11: Load saved model and predict
import joblib
import numpy as np

model = joblib.load('linear_model.pkl')
new_data = np.array([[0.03, 0.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]])
prediction = model.predict(new_data)
print("Predicted Price:", prediction)
