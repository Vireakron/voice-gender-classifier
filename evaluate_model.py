from data_processing import load_and_preprocess_data
from tensorflow.keras.models import load_model

X_train, X_test, y_train, y_test = load_and_preprocess_data()
model = load_model("models/voice_gender_model.keras")


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")