# 🧱 CNN Crack Detection Model (TensorFlow + OpenCV)

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to detect **cracks in surface images** (e.g., buildings, roads). It uses OpenCV for image loading and preprocessing. The model performs **binary classification**: cracked (defected) vs. non-cracked (non-defected) images.

## 📁 Project Structure
CNN_Crack_Detection/
├── CNN_crack_detection_model.keras # Trained model file
├── Crack_Detection_Model.ipynb # Jupyter Notebook (Colab compatible)
├── sample_image.jpg # Example test image
└── README.md 


---

## 📦 Features

- ✅ Image loading and resizing using OpenCV
- ✅ RGB conversion and normalization
- ✅ CNN model with 3 convolutional layers
- ✅ Binary classification using sigmoid activation
- ✅ Model saving and prediction on new images

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Google Colab

---

## 🧠 Model Architecture

Pyhton+tensorflow:

```python
keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])


🖼️ Sample Prediction Code:
from tensorflow import keras
import cv2
import numpy as np

# Load trained model
model = keras.models.load_model('CNN_crack_detection_model.keras')

# Load and preprocess image
img = cv2.imread('sample_image.jpg')
img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
print("🔴 Crack Detected" if pred[0][0] > 0.5 else "🟢 No Crack Detected")

📊 Results
Metric	Value
Accuracy	~99%
Loss	~0.0001
Input Size	128x128



