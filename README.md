# 🧠 Brain Tumor Classification Using Deep Learning

## 📌 Overview

This project focuses on **brain tumor classification from MRI images using Deep Learning**.

Two separate transfer learning approaches are implemented using **VGG16** and **ResNet18**, pretrained on the ImageNet dataset. Each model is trained independently to classify brain MRI images into four categories.

The predictions from VGG16 and ResNet18 are **not combined**. Each model is treated as an independent classification model and can be evaluated separately.

The project demonstrates the complete workflow of an image classification system, including:

* MRI image preprocessing
* Data augmentation
* Transfer learning
* VGG16 model training
* ResNet18 model training
* Model evaluation
* Performance visualization
* Brain tumor prediction
* Web-based prediction using Flask

> **Note:** This project is intended for research and educational purposes and should not be used as a substitute for professional medical diagnosis.

---

## 🎯 Objective

The primary objective is to develop deep learning models capable of automatically classifying brain MRI images into different tumor categories.

The project implements **VGG16 and ResNet18 separately** and evaluates each model independently for the same four-class classification task.

The system classifies MRI images into:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The trained model can also be integrated into a web application to provide predictions for new MRI images.

---

## 🧩 Tumor Classes

The models classify MRI images into four classes:

1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor**

---

# 🏗️ Model Architecture

Two separate deep learning architectures are used in this project:

* **VGG16**
* **ResNet18**

They are trained and evaluated **independently**.

### Overall Training Architecture

```text
                    Brain MRI Dataset
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
                   Data Augmentation
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
       VGG16 Model                 ResNet18 Model
       Training                    Training
             │                           │
             ▼                           ▼
       VGG16 Model                 ResNet18 Model
             │                           │
             ▼                           ▼
      Independent                 Independent
      Evaluation                  Evaluation
             │                           │
             ▼                           ▼
       VGG16 Prediction            ResNet18 Prediction
```

**The two models do not share predictions or combine their outputs.**

---

# 🧠 VGG16

VGG16 is a convolutional neural network architecture developed by the Visual Geometry Group.

The project uses VGG16 through **transfer learning**. A VGG16 model pretrained on ImageNet is used as the base network, and the classification component is adapted for the four brain tumor classes.

The model uses an input image size of:

```text
224 × 224 × 3
```

The trained VGG16 model is also integrated into the Flask web application for MRI image prediction.

---

# 🧠 ResNet18

ResNet18 is a convolutional neural network architecture based on **residual connections**.

Residual connections help information and gradients flow through the network and make the architecture effective for image classification.

ResNet18 is trained separately from VGG16 using the same brain tumor classification task.

The ResNet18 model has its own:

* Training process
* Validation process
* Evaluation
* Predictions

Its predictions are **not combined with VGG16 predictions**.

---

# 🔄 Transfer Learning

Transfer learning is used to adapt pretrained deep learning models to the brain MRI classification task.

The general workflow is:

```text
Pretrained ImageNet Model
          ↓
Use Pretrained Feature Extraction
          ↓
Adapt Classification Layer
          ↓
Train on Brain MRI Dataset
          ↓
Four-Class Classification
```

Transfer learning provides several advantages:

* Reduces training time
* Reuses pretrained visual features
* Requires less training data than training from scratch
* Helps the model converge faster
* Provides a strong baseline for image classification

---

# 🛠️ Technologies Used

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| **Python**       | Programming language    |
| **TensorFlow**   | Deep learning framework |
| **Keras**        | Neural network API      |
| **VGG16**        | Transfer learning model |
| **ResNet18**     | Transfer learning model |
| **NumPy**        | Numerical operations    |
| **Pandas**       | Data handling           |
| **Matplotlib**   | Visualization           |
| **Scikit-learn** | Model evaluation        |
| **Pillow**       | Image processing        |
| **Flask**        | Web application backend |

---

# 📂 Project Structure

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
│   ├── VGG16_brain_tumor_classification.ipynb
│   └── ResNet18_brain_tumor_classification.ipynb
│
├── models/
│   ├── VGG_16_brain_tumor.keras
│   └── ResNet18_brain_tumor.keras
│
├── images/
│   ├── confusion_matrix_vgg16.png
│   ├── confusion_matrix_resnet18.png
│   ├── vgg16_training_accuracy.png
│   ├── vgg16_training_loss.png
│   ├── resnet18_training_accuracy.png
│   └── resnet18_training_loss.png
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── app.py
├── predict.py
├── requirements.txt
└── README.md
```

> Update the filenames and folders according to your actual project structure.

---

# ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/brain-tumor-classification.git
```

### 2. Move into the project directory

```bash
cd brain-tumor-classification
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment on Windows

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

Example `requirements.txt`:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
pillow
flask
python-dotenv
groq
```

---

# 🔄 Data Preprocessing

MRI images are preprocessed before being passed to the deep learning models.

The preprocessing pipeline includes:

1. Loading MRI images
2. Converting images into the required format
3. Resizing images to **224 × 224 pixels**
4. Converting images into numerical arrays
5. Normalizing pixel values
6. Applying data augmentation to training images

Example augmentation techniques include:

* Rotation
* Horizontal flipping
* Zoom
* Translation

Data augmentation increases the diversity of training samples and can help the models generalize better to unseen MRI images.

---

# 🧠 VGG16 Training

VGG16 is trained independently using transfer learning.

The pretrained VGG16 feature extraction layers are used to learn visual representations from MRI images, while a classification head is adapted for the four brain tumor classes.

General workflow:

```text
Brain MRI Images
       ↓
Preprocessing
       ↓
Data Augmentation
       ↓
Pretrained VGG16
       ↓
Feature Extraction
       ↓
Classification Head
       ↓
4-Class Prediction
```

---

# 🧠 ResNet18 Training

ResNet18 is trained separately from VGG16.

General workflow:

```text
Brain MRI Images
       ↓
Preprocessing
       ↓
Data Augmentation
       ↓
Pretrained ResNet18
       ↓
Feature Extraction
       ↓
Classification Head
       ↓
4-Class Prediction
```

The ResNet18 training process is independent of the VGG16 training process.

---

# 📊 Model Evaluation

Each model is evaluated **separately**.

The following metrics can be used to evaluate the models:

### Accuracy

Measures the percentage of correctly classified MRI images.

### Precision

Measures how many images predicted as a particular class actually belong to that class.

### Recall

Measures how many actual images from a particular class are correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

A confusion matrix provides a detailed view of the classification performance for each class.

```text
                       Predicted
                Glioma  Meningioma  Pituitary  No Tumor

Actual Glioma

Actual Meningioma

Actual Pituitary

Actual No Tumor
```

Separate confusion matrices can be generated for:

```text
VGG16
  ↓
VGG16 Confusion Matrix


ResNet18
  ↓
ResNet18 Confusion Matrix
```

---

# ⚖️ Independent Model Comparison

VGG16 and ResNet18 are two **separate approaches** to the same classification problem.

They are not used as an ensemble.

```text
                    Brain MRI
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
           VGG16                ResNet18
             │                     │
             ▼                     ▼
        Prediction A          Prediction B
             │                     │
             ▼                     ▼
        Evaluated              Evaluated
        Separately             Separately
```

The performance of each model can be analyzed independently using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

---

# 🔍 Prediction

A trained model can be used to classify a new MRI image.

The general prediction workflow is:

```text
New MRI Image
      ↓
Image Validation
      ↓
Resize to 224 × 224
      ↓
Image Preprocessing
      ↓
Selected Trained Model
      ↓
Class Probabilities
      ↓
Highest Probability
      ↓
Predicted Class
```

For example:

```text
Predicted Class: Glioma
Confidence: 94.2%
```

The displayed confidence represents the probability assigned by the model to the predicted class. It should not be interpreted as medical certainty.

---

# 🌐 Flask Web Application

The project includes a **Flask-based web application** for serving the trained VGG16 model.

The Flask application provides a web interface where users can upload an MRI image and obtain a classification prediction.

### Web Application Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │  Web Interface  │
                  │    index.html   │
                  └────────┬────────┘
                           │
                           │ HTTP Request
                           ▼
                  ┌─────────────────┐
                  │      Flask      │
                  │     Backend     │
                  └────────┬────────┘
                           │
                           ▼
                  Image Validation
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
                    Trained VGG16
                           │
                           ▼
                  Class Probabilities
                           │
                           ▼
                    Predicted Class
                           │
                           ▼
                  Prediction + Probability
                           │
                           ▼
                  ┌─────────────────┐
                  │  Web Interface  │
                  └─────────────────┘
```

The Flask application currently loads the trained VGG16 model and uses it for inference.

---

# 🔗 Flask Routes

The Flask application contains routes for different operations.

| Route      | Method | Purpose                                          |
| ---------- | ------ | ------------------------------------------------ |
| `/`        | GET    | Displays the home page                           |
| `/predict` | POST   | Receives MRI image and performs VGG16 prediction |
| `/ask`     | POST   | Sends user question to the AI assistant          |
| `/clear`   | GET    | Clears stored prediction/session data            |

The `/predict` route receives the uploaded image, validates the file, temporarily saves it, preprocesses it, performs prediction using the trained VGG16 model, stores the prediction and confidence, and removes the temporary image.

---

# 🤖 AI Educational Assistant

The web application also includes an AI-based educational assistant.

Users can ask general questions related to:

* Brain tumors
* Glioma
* Meningioma
* Pituitary tumors
* MRI
* Medical terminology
* General information about the predicted class

The Flask application provides the current VGG16 prediction and model probability as context to the AI assistant.

### AI Architecture

```text
User Question
      ↓
Flask /ask Route
      ↓
Current VGG16 Prediction
      ↓
Model Probability
      ↓
Groq API
      ↓
LLM
      ↓
Educational Response
      ↓
Flask
      ↓
Web Interface
```

The AI assistant is separate from the deep learning classification model.

```text
VGG16  → MRI Image Classification

Groq   → Text-Based Educational Assistant
```

---

# 🔐 Environment Variables

Sensitive configuration such as the Groq API key and Flask secret key is loaded using environment variables.

Example:

```text
GROQ_API_KEY=your_api_key
FLASK_SECRET_KEY=your_secret_key
```

The application uses `python-dotenv` to load these variables.

This avoids directly placing the API key inside the source code.

---

# 🗑️ Temporary Image Handling

Uploaded MRI images are temporarily stored during the prediction process.

The workflow is:

```text
Upload MRI
    ↓
Temporary Storage
    ↓
Preprocessing
    ↓
VGG16 Prediction
    ↓
Delete Temporary Image
```

The Flask application removes the uploaded image after prediction.

This prevents unnecessary accumulation of uploaded MRI images.

---

# 💡 Key Learning Outcomes

Through this project, the following concepts are demonstrated:

* Convolutional Neural Networks
* Transfer Learning
* VGG16
* ResNet18
* Image preprocessing
* Data augmentation
* Multi-class classification
* Independent model training
* Independent model evaluation
* Confusion matrix
* Precision
* Recall
* F1-score
* Model comparison
* Flask web application development
* Machine learning model deployment
* API integration

---

# 🔮 Future Improvements

Possible future improvements include:

* Compare VGG16 and ResNet18 with additional architectures such as ResNet50, EfficientNet, and DenseNet
* Use Grad-CAM for model explainability
* Perform hyperparameter tuning
* Address class imbalance
* Increase dataset size
* Improve model calibration
* Experiment with different data augmentation techniques
* Deploy the best-performing model using a production server
* Build a REST API for model prediction
* Add explainable AI for medical image analysis

---

# ⚠️ Disclaimer

This project is developed for **educational and research purposes only**.

The predictions generated by the models should **not be considered medical advice or a clinical diagnosis**.

Brain tumor diagnosis should be performed by qualified medical professionals using appropriate clinical and diagnostic procedures.

---

# 👨‍💻 Author

**Ajay Vardhan Naik**

GitHub: `https://github.com/your-username`

---

## ⭐ If You Find This Project Useful

If you found this project useful for learning deep learning, transfer learning, medical image classification, or machine learning deployment, consider giving the repository a ⭐.
