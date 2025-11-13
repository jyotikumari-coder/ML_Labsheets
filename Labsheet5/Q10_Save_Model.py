# Q10: Save trained model using joblib
import joblib
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

joblib.dump(model, 'linear_model.pkl')
print("Model saved successfully as linear_model.pkl")
