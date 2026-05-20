import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved model and data
model = load_model('speech_emotion_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nSpeech Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))