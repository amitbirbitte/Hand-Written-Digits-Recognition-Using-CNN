# 🧠 Handwritten Digit Recognition using CNN

![Handwritten Digit Recognition](cover-image.png)

> A deep learning project to classify handwritten digits (0–9) from the MNIST dataset using Convolutional Neural Networks (CNNs).

---

## 📌 Project Overview

This project builds and compares deep learning models to recognize handwritten digits from the popular **MNIST** dataset.  
The main goal is to design a robust CNN-based classifier, evaluate multiple architectures, select the best-performing model, and enable prediction on new handwritten digit images after deployment.

---

## 🎯 Objectives

- Load and explore the MNIST handwritten digits dataset.
- Preprocess image data for use in CNN models.
- Build and train **multiple CNN models** (Simple CNN and Deeper CNN).
- Compare model performance using accuracy, loss, confusion matrix, and classification report.
- Save the best model and use it for **real-time predictions** on new images.

---

## 🧩 Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Data Handling & Analysis:** NumPy, Pandas (optional), Matplotlib, Seaborn  
- **Model Evaluation:** scikit-learn (classification report, confusion matrix)  
- **Environment:** Jupyter Notebook

---

## 📂 Project Structure (Example)

```bash
.
├── Handwritten-Digits-Recognition.ipynb   # Main notebook (all tasks)
├── hand_written_digits_prediction.h5      # Saved best CNN model
├── digit_examples/                        # Sample test images for prediction
├── cover-image.png                        # Cover image for README
└── README.md                              # Project documentation
```

> 🔁 All tasks (EDA, model building, comparison, challenges, and prediction) are completed inside **a single Jupyter notebook**, as per project requirements.

---

## 📊 Dataset Description – MNIST

- **Total images:** 70,000  
  - 60,000 for training  
  - 10,000 for testing  
- **Image size:** 28 × 28 pixels  
- **Channels:** 1 (grayscale)  
- **Classes:** 10 (digits 0–9)

Each image is a handwritten digit and is labeled with the correct digit, which makes this a multi-class classification problem.

---

## 🏗️ Model Architectures

### 1️⃣ Simple CNN

A lightweight CNN with:

- 2 × Conv2D + ReLU
- 2 × MaxPooling2D
- Flatten
- Dense(128, ReLU)
- Dense(10, Softmax)

✅ Good baseline model  
✅ Fast training, simple architecture

---

### 2️⃣ Deeper CNN

A more powerful CNN with:

- Multiple Conv2D layers
- MaxPooling2D after convolution blocks
- Dropout layers for regularization
- Dense(256, ReLU)
- Dense(10, Softmax)

✅ Higher accuracy  
✅ Better generalization  
⚠️ Slightly higher training time

---

## 📈 Model Comparison

Two CNN models were trained and evaluated on the same MNIST data.  
The **Deeper CNN** achieved slightly higher test accuracy and showed better generalization across digit classes, especially for difficult digits such as 4, 5, 8, and 9.  
Therefore, the Deeper CNN model is recommended for production use, while the Simple CNN can be used as a lightweight fallback model for low-resource environments.

---

## 🔁 Training Workflow (Notebook Steps)

1. **Import Libraries** (TensorFlow, NumPy, Matplotlib, etc.)
2. **Load Dataset** using `tf.keras.datasets.mnist`.
3. **Exploratory Data Analysis (EDA)**  
   - View dataset shapes  
   - Visualize sample digits  
   - Analyze label distribution
4. **Preprocessing**  
   - Normalize pixel values to `[0, 1]`  
   - Reshape to `(28, 28, 1)`  
   - One-hot encode labels for Softmax
5. **Build CNN Models** (Simple CNN & Deeper CNN).
6. **Compile & Train Models** using `Adam` optimizer and `categorical_crossentropy` loss.
7. **Evaluate Models** on test data.
8. **Model Comparison** (accuracy, loss, confusion matrix, classification report).
9. **Save Best Model** as `.h5`.
10. **Test Prediction** on custom handwritten digit images.

---

## 🤖 How to Run the Project

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/handwritten-digits-cnn.git
cd handwritten-digits-cnn
```

2. **Create and Activate Virtual Environment** (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
# source venv/bin/activate  # on macOS / Linux
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Open Jupyter Notebook**

```bash
jupyter notebook Handwritten-Digits-Recognition.ipynb
```

5. Run all cells in order to:
   - Train models  
   - Compare performance  
   - Save the best model  
   - Test prediction

---

## 🔮 Prediction After Deployment

Once the best model is saved (e.g., `hand_written_digits_prediction.h5`), you can load it and predict new digits:

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model("hand_written_digits_prediction.h5")

# Load and preprocess custom digit image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = 255 - img            # invert if needed to match MNIST
img = img.astype("float32") / 255.0
img = img.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)
print("Predicted Digit:", predicted_digit)
```

---

## 🚧 Challenges Faced

- **Similar-looking digits** (e.g., 3 vs 5, 4 vs 9) caused misclassifications.
- **Overfitting risk** in deeper networks with many parameters.
- **Training time** increased with deeper models on CPU environments.

These were addressed using:

- Dropout layers for regularization  
- Validation split to monitor performance  
- Proper normalization and consistent preprocessing  
- Testing several architectures and choosing the best-performing one

---

## 📌 Future Improvements

- Add **data augmentation** (shifts, rotations, zoom) to improve robustness.
- Deploy as a **web app** using Streamlit or Flask.
- Extend the model to recognize other handwritten symbols or multi-digit numbers.
- Convert model to TensorFlow Lite for use on **mobile devices**.

---

## ✍️ Author

**Name:** Amit Birbitte  
**Role:** Data Science / Machine Learning Enthusiast  
**Project:** Handwritten Digit Recognition using Deep Learning (CNN)

---

## 🖼️ About the Cover Image

The `cover-image.png` file is used at the top of this README as a visual banner.  
You can create your own cover image (e.g., using Canva, Figma, or PowerPoint), save it as `cover-image.png`, and place it in the project root so that it appears correctly in the GitHub README.

---

💡 *Feel free to fork, modify, and extend this project for learning, experimentation, or portfolio use.*  
If you build something cool on top of it, don’t forget to share!
