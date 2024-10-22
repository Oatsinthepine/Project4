# Project4

This repo is for project 4

## Topic (Health): Brain Tumor Detection using tensorflow CNN

This project is aiming for constructing a deep-neural network model using CNN to detect brain tumor. The data source is from Kaggle website `Brain Tumor Classification MRI`. 

The approach for the project are divided in the steps below:

* Download the source image data from the kaggle url. Perfrom train/test data resize due to large original data size. Considering the time limitation this project will limit the number of data used for training the model.

* Model building and training. Which include hyperparameter tunning and model saving in h5 format.

* Hosting trained model using Flask and construct a simple user interface that enable users to upload random brain MRI picture, then the model will predict and give users some prediction based on the image provided.

## New knowledege required for this project:

First we need to learn CNN and it's structure. Also Keras provide some pre-training model specifically for image classification. e.g: Xception,SVC, HOG, VGG16, VGG19 etc. 

Then when tunning, consider Keras-tuner. Save the history and all the hyperparameters, convert the training loss, accuracy etc into a pd.Dataframe for comparison.


## Data Source:

Bhuvaji, S. Kadam, A. Bhumkar, P. Dedge, D. Chakrabarty, N. Kanchan, S. (2020). Brain Tumor Classification (MRI) [Data set]. Kaggle. <https://doi.org/10.34740/KAGGLE/DSV/1183165>

