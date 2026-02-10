# paste evaluation code here
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import datasets
from sklearn.metrics import confusion_matrix, classification_report

class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# -----------------------
# Load model
# -----------------------
model = tf.keras.models.load_model("improved_cifar10_model.keras")

# -----------------------
# Load test data
# -----------------------
(_, _), (x_test, y_test) = datasets.cifar10.load_data()

x_test = x_test / 255.0
y_true = y_test.flatten()

# -----------------------
# Predictions
# -----------------------
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------
# Accuracy
# -----------------------
accuracy = np.mean(y_pred == y_true)
print("Test Accuracy:", accuracy)

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------
# Classification Report
# -----------------------
print(classification_report(y_true, y_pred, target_names=class_names))
