# Project4 Group 2

## Topic (Health): Brain Tumor classification and detection using deep-learning models

This project is aiming for finding the best performing model to classifiy different types of brain tumors by using different deep-learning models. Also the best perfroming model will be used to be deployed on Flask server (localhost) for users to test by uploading the image files, and model will returns per predictions in percentage.

Team members:

- Ziyue Zhou 
    * Group Leader: Responsible for cnn model building, training & optimisation, Flask app build, making slide

- Kenneth Le
    * Contributor: Responsible for Xception model training & optimisation, assist Flask app build, making slide

- Singh Rakhi
    * Contributor: Responsible for ResNet50 model training & optimisation, model history plotting for the whole group, making slide

- Yuqi Huang
    * Contributor: Responsible for U-NET model (classifcation & segementation) training & optimisation, making slide

- Eric Tran
    * Contributor: Testing deep-learning models and implemented flask_app in different method, making slides

The data source is from Kaggle website `Brain Tumor Classification MRI`. 

## Approach for the project

The approach for the project are listed in the steps below:

* Download the source image data from the kaggle url. Perfroming data preprocessing (which include data loading, resizing, rescaling, normalising, augumenting) for the train/test/validate splited data subset.

* Deep-learning model building and training from the preprocessed training data. Testing and evaluating the model performance against the testing dataset. 

* Next is model optimisation, start from simple approach including increase more epochs, adjust learning rate, add schedular, checkpoint & callbacks. Then re-evaluate performance. Then try advanced optimisation methods like Keras-tuner and hyperparameter tunning. Saving the best performing model in h5 format.

* Hosting the best performing model using Flask, and construct a simple user interface that enable users to upload sample brain MRI image to test by themselves, the model will predict and give users predictions based on the image provided.

## Instructions for using the project

To see each model's training & optimising details, go to the folder named per group member in the `main-branch`. it contains all source files and scripts for per model training.
Please note that all final works are cleaned and packed in the `main-branch`.
To test the best performing model and experience the prediction using webpage interface. Please use git pull / or download the flask_apps folder to your local. Then, please run the app.py file via `flask run` to start the server. Then follow the instruction given in the script to access the predict.html webpage to uplaod images and see the test prediction.

## Data privacy & ethical considerations

The original source data did not contains any personal identifiable information (PII). Also, the dataset has licence MIT. Which this project only use the data for educational and practical purpose only.


## Data Source:

Bhuvaji, S. Kadam, A. Bhumkar, P. Dedge, D. Chakrabarty, N. Kanchan, S. (2020). Brain Tumor Classification (MRI) [Data set]. Kaggle. <https://doi.org/10.34740/KAGGLE/DSV/1183165>

## References

Deploy Keras Neural Network to Flask Web Service, deeplizard, <https://www.youtube.com/watch?v=SI1hVGvbbZ4&list=PLZbbT5o_s2xr34kj-vyrIXzvUJsG3z5S_>

Create CNN Model and Optimize using Keras Tuner, Krish Naik, <https://www.youtube.com/watch?v=OzLAdpqm35E&t=639s>

What are Callbacks, Checkpoints and Early Stopping in deep learning (Keras and TensorFlow), DigitalSreeni, <https://www.youtube.com/watch?v=wkwtIeq9Ljo>




