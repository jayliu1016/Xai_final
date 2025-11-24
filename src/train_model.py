# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

def load_data():
    
    df = pd.read_csv("data/spam.csv", encoding="latin-1")

    
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df = df[df['label'].isin(['ham', 'spam'])]

    return df

def train_model():
    df = load_data()
    X = df['text']
    y = df['label'].map({'ham': 0, 'spam': 1})

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("logreg", LogisticRegression(max_iter=1000, n_jobs=-1)),
    ])

    clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/spam_model.joblib")
    print(" save：models/spam_model.joblib")

if __name__ == "__main__":
    train_model()
