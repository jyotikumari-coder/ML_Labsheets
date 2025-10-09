"""
Problem 2: Fake News Classification
Models: Logistic Regression, Decision Tree, SVM
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # Load dataset
    df = pd.read_csv("datasets/fake_or_real_news.csv")  # Ensure dataset path
    
    # Features and target
    X = df["text"]
    y = df["label"].map({"FAKE": 0, "REAL": 1})  # Encode labels
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Text preprocessing with TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=100, random_state=42),
        "SVM (Linear Kernel)": SVC(kernel="linear")
    }
    
    # Train and evaluate
    for name, model in models.items():
        print(f"\n===== {name} =====")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
