# SkinGuard-AI
SkinGuard AI is an AI-powered web application that helps in the early screening of skin cancer by analyzing skin lesion images.

## 🔬 Model Training

The model is trained on a binary image dataset with two classes: **Benign** and **Malignant**.

- Input size: 224x224
- Framework: TensorFlow / Keras
- Model type: CNN (2 Conv layers + Dense)
- Activation: ReLU, Sigmoid
- Loss: Binary Crossentropy
- Optimizer: Adam

Training script is available at: [`model_training.py`](model_training.py)
