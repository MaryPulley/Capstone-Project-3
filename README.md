# Project 3-Garbage Classification Model

## Team 6
Asa Adomatis, Mary Pulley, Vikram Borra, Sophak So, Katie Craig
TA- Rimi Sharma

## Overview

This project aims to develop an image classification model using a convolutional neural network (CNN) architecture to determine whether an object is recyclable. The model classifies images into one of 8 categories, each corresponding to a type of recyclable material. By leveraging machine learning and computer vision, this project provides users with a quick decision-making tool to promote recycling.


## Instructions to Run Code
To use User Interface Application - "Waste Classifier App" Download the following files to utilize with the UI_code_final.ipynb notebook

garbage_classification.csv

garbage_recycle_model.pkl

preprocessing.py

Folder EXAMPLE IMAGES

## Package Install

1. bash pip install tensorflow
2. pip install pandas
3. pip install Pillow
4. pip install scikit-learn
5. pip install kagglehub

## Dataset

The dataset used for training and evaluation is the "Garbage Classification" dataset available on Kaggle:

**Dataset source:** [https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data)

This dataset contains images of various types of garbage, such as paper, plastic, glass, metal, and cardboard. The images are labeled with their corresponding garbage categories.

## Data Preprocessing

1. **Data Collection:** Gather a dataset of images containing various types of garbage, labeled as "recyclable" or "garbage". You can use public datasets or create your own by taking pictures of different waste items.
2. **Data Cleaning:** Initially, the dataset classified the data into 12 categories.  However, we found it more effcient to reduce the number of categories to 8 by combining a few that were closely related. 
3. **Data Augmentation:** Apply data augmentation techniques (e.g., rotation, flipping, scaling) 
4. **Data Splitting:** Divide the dataset into training, validation, and testing sets. This allows for proper model evaluation and prevents overfitting.

## Model Selection

- **Convolutional Neural Networks (CNNs):** CNNs are widely used for image classification tasks and have proven to be effective in identifying patterns and features in images. Pre-trained models like ResNet, Inception, or MobileNet can be used as a starting point or fine-tuned for this specific task.
- Specific  CNN Model (Sequential model): Is the model we used with multiple convolutional layers, pooling layers, and fully connected layers.  The model is trained using the Adam optimizer and categorical cross-entropy loss function. 


## Model Training

1. **Preprocessing:** Resize and normalize the images to a consistent format.
2. **Model Initialization:** Instantiate the chosen model with appropriate hyperparameters.
3. **Training:** Train the model on the training dataset using an optimizer (e.g., Adam, SGD) and a loss function (e.g., categorical cross-entropy). Monitor the training progress and adjust hyperparameters as needed.
4. **Validation:** Evaluate the model's performance on the validation set during training to prevent overfitting and ensure generalization.

## Model Testing

1. **Evaluation:** Evaluate the trained model on the testing dataset to assess its performance on unseen data.
2. **Metrics:** Use appropriate metrics like accuracy, precision, recall, and F1-score to measure the model's performance.
3. **Analysis:** Analyze the results and identify areas for improvement.

## Usage

1. Download the dataset from the link above.
2. Install the required packages.
3. Run the code to train the model.
4. Use the link to gradio (UI_code_final.ipynb) 

## Deployment

1. **Model Saving:** Save the trained model for future use.
2. **Platform:** We chose Gradio as our U/I 
3. **Integration:** Integrate the model into the chosen platform, allowing users to input images or video streams for classification.


