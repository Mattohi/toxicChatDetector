import joblib
import pandas as pd
import os

# Load model
model = joblib.load("toxic_model.pkl")

print("=== TOXIC CHAT DETECTOR ===")
print("Ketik 'exit' untuk keluar.\n")

while True:

    text = input("Chat: ")

    if text.lower() == "exit":
        break

    # Prediksi
    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]

    if prediction == 1:
        label = "TOXIC"
        confidence = probability[1] * 100
    else:
        label = "NON-TOXIC"
        confidence = probability[0] * 100

    print("\nResult:", label)
    print("Confidence:", round(confidence, 2), "%")

    # Tanya apakah prediksi benar
    correct = input("Prediksi ini benar? (y/n): ").lower()

    if correct == "n":

        # Tanya label yang sebenarnya
        while True:
            actual = input("Seharusnya apa? (toxic/non-toxic): ").lower()

            if actual == "toxic":
                true_label = 1
                break

            elif actual == "non-toxic":
                true_label = 0
                break

            else:
                print("Masukkan 'toxic' atau 'non-toxic'.")

        # Tambahkan ke dataset
        new_data = pd.DataFrame({
            "text": [text],
            "label": [true_label]
        })

        dataset_path = "dataset/toxic.csv"

        new_data.to_csv(
            dataset_path,
            mode="a",
            header=not os.path.exists(dataset_path),
            index=False
        )

        print("\nData berhasil ditambahkan ke dataset!")
        print("Text:", text)
        print("Label:", true_label)

    else:
        print("Oke, prediksi dianggap benar.")

    print()