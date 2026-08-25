# 🧠 Brain Tumor Classification Using Deep Learning

## 📌 Overview

This project focuses on **brain tumor classification from MRI images using Deep Learning**.

A transfer learning approach is used with the **VGG16 convolutional neural network**, pretrained on ImageNet. The pretrained convolutional layers are used as a feature extractor, while a new classification head is trained to classify brain MRI images into four tumor categories.

The project demonstrates the complete workflow of an image classification system, including:

* MRI image preprocessing
* Data augmentation
* Transfer learning
* Model training
* Model evaluation
* Performance visualization
* Brain tumor prediction

---

## 🎯 Objective

The primary objective is to develop a deep learning model capable of automatically classifying brain MRI images into different tumor categories.

The model can assist in the **automated analysis of MRI images** and serve as a research/educational demonstration of deep learning for medical image classification.

> **Note:** This project is intended for research and educational purposes and should not be used as a substitute for professional medical diagnosis.

---

## 🧩 Tumor Classes

The model classifies MRI images into **four classes**:

1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

---

## 🏗️ Model Architecture

The project uses **VGG16 Transfer Learning**.

### Architecture

```text
Input MRI Image
       ↓
Image Resizing (224 × 224)
       ↓
Image Preprocessing
       ↓
VGG16 Pretrained Base
       ↓
Feature Extraction
       ↓
Global Average Pooling / Flattening
       ↓
Fully Connected Layers
       ↓
Output Layer
       ↓
4-Class Classification
```

### Why VGG16?

VGG16 is a well-known convolutional neural network architecture that has been pretrained on the ImageNet dataset.

Using transfer learning provides several advantages:

* Reduces training time
* Requires less training data
* Reuses pretrained visual features
* Improves convergence
* Provides a strong baseline for image classification

---

## 🛠️ Technologies Used

| Technology   | Purpose                 |
| ------------ | ----------------------- |
| Python       | Programming language    |
| TensorFlow   | Deep learning framework |
| Keras        | Neural network API      |
| VGG16        | Transfer learning model |
| NumPy        | Numerical operations    |
| Pandas       | Data handling           |
| Matplotlib   | Visualization           |
| Scikit-learn | Model evaluation        |

---

## 📂 Project Structure

```text
Brain-Tumor-Classification/
│
├── dataset/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── notebooks/
│   └── brain_tumor_classification.ipynb
│
├── models/
│   └── brain_tumor_model.keras
│
├── images/
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   └── training_loss.png
│
├── requirements.txt
├── README.md
└── predict.py
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/brain-tumor-classification.git
```

Move into the project directory:

```bash
cd brain-tumor-classification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Example `requirements.txt`:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
pillow
```

---

## 🔄 Data Preprocessing

MRI images are preprocessed before being passed to the model.

The preprocessing pipeline includes:

1. Loading MRI images
2. Resizing images to **224 × 224 pixels**
3. Converting images into numerical arrays
4. Normalizing/preprocessing pixel values
5. Applying data augmentation to training images

Example augmentation techniques include:

* Rotation
* Horizontal flipping
* Zoom
* Translation

Data augmentation helps the model generalize better to unseen MRI images.

---

## 🧠 Transfer Learning

The project uses:

```python
from tensorflow.keras.applications import VGG16
```

The VGG16 model is loaded with ImageNet pretrained weights:

```python
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
```

The original ImageNet classification layer is removed using:

```python
include_top=False
```

A new classification layer is then added for the four brain tumor classes.

---

## 🚀 Training

The model is trained using the prepared MRI dataset.

During training, the following metrics can be monitored:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

Training and validation curves can be plotted to analyze whether the model is learning effectively or overfitting.

---

## 📊 Model Evaluation

The model can be evaluated using several classification metrics.

### Accuracy

Measures the percentage of correctly classified MRI images.

### Precision

Measures how many predicted samples of a particular class are actually correct.

### Recall

Measures how many actual samples of a class are correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

A confusion matrix provides a detailed view of how images from each tumor class are classified.

Example:

```text
                    Predicted
              Glioma  Meningioma  Pituitary  No Tumor

Actual Glioma
Actual Meningioma
Actual Pituitary
Actual No Tumor
```

---

## 📈 Results

Add your final results here after training the model.

Example:

| Metric              | Score |
| ------------------- | ----: |
| Training Accuracy   |   XX% |
| Validation Accuracy |   XX% |
| Test Accuracy       |   XX% |
| Precision           |   XX% |
| Recall              |   XX% |
| F1-Score            |   XX% |

> Replace the `XX%` values with the actual results from your trained model.

---

## 🔍 Prediction

The trained model can be used to predict the class of a new MRI image.

Example workflow:

```text
New MRI Image
      ↓
Resize to 224 × 224
      ↓
Preprocess Image
      ↓
Load Trained VGG16 Model
      ↓
Generate Prediction
      ↓
Predicted Tumor Class
```

Example output:

```text
Predicted Class: Glioma
Confidence: 94.2%
```

---

## 💡 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Convolutional Neural Networks
* Transfer Learning
* VGG16 architecture
* Image preprocessing
* Data augmentation
* Multi-class classification
* Model evaluation
* Confusion matrix
* Precision, Recall and F1-score
* Medical image classification

---

## 🔮 Future Improvements

Possible improvements include:

* Compare VGG16 with ResNet50, EfficientNet and DenseNet
* Use Grad-CAM for model explainability
* Perform hyperparameter tuning
* Handle class imbalance
* Increase dataset size
* Use ensemble learning
* Deploy the model as a web application
* Build a REST API for prediction
* Add explainable AI for medical image analysis

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The predictions generated by this model should **not be considered medical advice or a clinical diagnosis**. Brain tumor diagnosis should be performed by qualified medical professionals using appropriate clinical and diagnostic procedures.

---

## 👨‍💻 Author

**Ajay Vardhan Naik**

GitHub: `https://github.com/your-username`

---

## ⭐ If You Find This Project Useful

If you found this project helpful for learning deep learning, transfer learning, or medical image classification, consider giving the repository a ⭐.
