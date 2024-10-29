from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import cv2
import os
from werkzeug.utils import secure_filename
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'brain_tumor_classifier.h5'  # Update this to the correct path of your model file
model = load_model(MODEL_PATH)

# Configure the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Tumor categories (update these to match the categories your model was trained on)
categories = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction function
def model_predict(img_path, model):
    IMG_SIZE = (128, 128)  # Adjust this size to match your model's expected input size

    # Load and preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)  # Index of highest probability
    predicted_label = categories[predicted_class_index]  # Map index to category name
    probability = float(prediction[0][predicted_class_index])  # Probability of the predicted class

    return predicted_label, probability

# Route for home page and upload form
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename and save it
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the uploads folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            file.save(file_path)

            # Process and predict
            predicted_label, probability = model_predict(file_path, model)
            
            # Render the result template with values
            return render_template('result.html', filename=filename, label=predicted_label, probability=probability)
    
    # Display the upload form if the request method is GET
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
