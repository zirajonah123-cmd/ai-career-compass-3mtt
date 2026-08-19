import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/career_data.csv"
MODEL_PATH = "model.joblib"

df = pd.read_csv(DATA_PATH)

def combine(row):
    return " | ".join([
        str(row["education_level"]),
        str(row["field_of_study"]),
        str(row["skills"]),
        str(row["interests"]),
        str(row["career_goal"])
    ])

X = df.apply(combine, axis=1)
y = df["career"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))

joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
