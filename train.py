import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv("dataset/toxic.csv")

print("Jumlah data:", len(data))
print(data.head())

X = data["text"]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training selesai!")


prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAkurasi:", accuracy)

print("\nLaporan klasifikasi:")
print(
    classification_report(
        y_test,
        prediction,
        target_names=["Tidak Toxic", "Toxic"]
    )
)


joblib.dump(model, "toxic_model.pkl")

print("\nModel berhasil disimpan sebagai toxic_model.pkl")
