import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "data/spam.csv"
MODEL_FILE = "model/spam_model.pkl"

data = pd.read_csv(DATA_FILE)
X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

os.makedirs("model", exist_ok=True)
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_FILE}")
