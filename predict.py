import joblib
import pandas as pd
import os

model = joblib.load("toxic_model.pkl")

print("=== TOXIC CHAT DETECTOR ===")
print("Type 'exit' to quit.\n")

while True:

    text = input("Chat: ")

    if text.lower() == "exit":
        break

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

    correct = input("Was this prediction correct? (y/n): ").lower()

    if correct == "n":
   
        true_label = 1 - int(prediction)

        dataset_dir = "dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, "toxic.csv")

        new_data = pd.DataFrame({
            "text": [text],
            "label": [true_label]
        })

        new_data.to_csv(
            dataset_path,
            mode="a",
            header=not os.path.exists(dataset_path),
            index=False
        )

        print("\nData added to the dataset!")
        print("Text:", text)
        print("Label:", "TOXIC" if true_label == 1 else "NON-TOXIC")

    else:
        print("Okay, prediction marked as correct.")

    print()
