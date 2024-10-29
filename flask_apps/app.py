# get all dependencies

import base64
import io
from PIL import Image
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from flask_cors import CORS # this dependency is required to deal with the CORS issue
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5000"}}) # for security concerns, only allow the predict html page to enable CORS, for the project only.

@app.route('/')
def home_page():
    return "This is the home page. For using the model to predict your image, please visit: 'http://127.0.0.1:5000/predict'"

@app.route('/test')
def running():
    return "Flask is running!" # if you see this, them means the backend flask server is running normally.

def get_model():
    global model 
    model = load_model("model_checkpoint_epoch_177.h5") # change the path here for your local path when accessing this .h5 model file.
    print("model successfully loaded.") # when you run flask, if successful, you should see this message print in the terminal.

def preprocess_image(img, traget_size = (256, 256)):
    if img.mode != "L":
        img = img.convert("L") # L means grayscale
    img = img.resize(traget_size) # if user image is not in correct size, resize it here.
    img_array = img_to_array(img)

    # need to implement rescaling here, by dividing 255 to convert the numpyfied array, to reach the result between 0 and 1.
    img_array_scaled = img_array / 255.0

    img_array = np.expand_dims(img_array_scaled, axis=0)

    return img_array

print("==========>> Loading keras model...")

# invoke the model function and get the model object.
get_model()


@app.route("/tumor_predict", methods = ['GET', 'POST'])
def predict(): # here is the predict function for user 
    message = request.get_json(force=True)
    encoded = message['image'] # file passed in need to be image format
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, traget_size=(256, 256))

    # here use the model to predict the user uploaded image file
    prediction = model.predict(processed_image)
    # here, to directly access the predicted probabilities. use index slicing to reterive the result.
    # As model.predict() returns 2D np.array which first dim contains batch_size info. we don't need this.
    predicted_probabilities = prediction[0]

    # create a dictionary to holf all returned classes probabilities.
    # here the index co-respond to the classes.
    response = {
        'Prediction': {
            "glioma_tumor": round(float(predicted_probabilities[0]), 4),
            "meningioma_tumor": round(float(predicted_probabilities[1]), 4),
            "no_tumor": round(float(predicted_probabilities[2]), 4),
            #"pituitary_tumor": float(predicted_probabilities[3]) # U-Net model did not trained on this class, so comment it out
        }
    }

    return jsonify(response)

# here is the flask url for using the backend model to predict user uplaoded image file
@app.route("/predict")
def predict_html():
    # call the flask build-in method to render my predict.html webpage.
    return render_template("predict.html")



# use app.run to start flask server
if __name__ == "__main__":
    app.run(debug=True)