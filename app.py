from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}


# Folder and path
os.makedirs("model", exist_ok=True)
model_path = "model/skin_cancer.h5"

# Google Drive download (replace with your file ID)
file_id = "1j-6ez6DH1fupdQFawAg8ANJX7P03a6dr"  
url = f"https://drive.google.com/uc?id={file_id}"

# Download if model not already downloaded
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)


# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict using the model
        image_data = preprocess_image(filepath)
        prediction = model.predict(image_data)[0][0]
        result = 'Malignant' if prediction < 0.5 else 'Benign'
        confidence = float(prediction) if result == 'Malignant' else 1 - float(prediction)

        return render_template('analyze.html', result=result, confidence=confidence, image_path=filepath)

    return render_template('analyze.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
