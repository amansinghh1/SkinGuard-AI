# 🛡️ SkinGuard AI - Skin Cancer Detection Web App

SkinGuard AI is a deep learning-powered web application that provides **early screening** of skin cancer using image classification techniques. Built with **Flask**, **TensorFlow**, and a clean modern UI, this tool allows users to upload images of skin lesions and receive instant AI-based predictions.

---

## 🚀 Features

- 📷 Upload photo for analysis
- 🤖 Deep learning model (CNN) predicts: **Benign** or **Malignant**
- 📊 Displays confidence score
- ⚠️ Medical disclaimer for ethical guidance
- ✅ Tips for accurate image submission
- 📎 Sticky footer with important info
- 🎨 Transparent feature overlays and aesthetic landing UI

---

## 🧠 Model Information

- **Architecture**: Custom Convolutional Neural Network (CNN)
- **Trained on**: Augmented dataset of benign and malignant skin lesion images
- **Input shape**: 224x224 RGB
- **Loss**: Binary Cross-Entropy  
- **Framework**: TensorFlow/Keras  
- **Training**: Done in Google Colab using GPU  
- **File**: `skin_cancer.h5` (~127 MB) [Stored in Google Drive due to GitHub size limits]

👉 [📥 Download Model](https://your-google-drive-link.com) and place in: `model/skin_cancer.h5`

---

## 📁 Project Structure
<pre> ``` SkinGuard_Complete_Project/ ├── app.py # Main Flask application ├── train_model.py # Script to train the model (optional) ├── preprocess.py # Image preprocessing logic ├── requirements.txt # Python dependencies ├── README.md # Project documentation ├── .gitignore # Ignore list (model, venv, etc.) │ ├── model/ │ └── skin_cancer.h5 # Trained ML model (excluded from GitHub) │ ├── static/ │ ├── css/ │ │ └── style.css # Stylesheet for all pages │ │ │ ├── images/ │ │ └── bg.png # Background image │ │ │ └── uploads/ │ └── ... # User uploaded images (generated at runtime) │ ├── templates/ │ ├── index.html # Landing page │ └── analyze.html # Result page with predictions ``` </pre>
