import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

model = load_model('text_emotion_model.h5')
with open('label_encoder_text.pkl', 'rb') as f:
    le = pickle.load(f)

X_test = np.load('X_test_text.npy')
y_test = np.load('y_test_text.npy')

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nText Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))