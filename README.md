# Project 3-Garbage Classification Model

## Team 6
Asa Adomatis, Mary Pulley, Vikram Borra, Sophak So, Katie Craig
TA- Rimi Sharma

## Overview

This project aims to develop an image classification model using a convolutional neural network (CNN) architecture to determine whether an object is recyclable. The model classifies images into one of 12 categories, each corresponding to a type of recyclable material. By leveraging machine learning and computer vision, this project provides users with a quick decision-making tool to promote recycling.


## Package Install

pip install tensorflow
pip install keras
pip install Pillow
pip install request
pip install Gradio


## Data Preprocessing

1. **Data Collection:** Gather a dataset of images containing various types of garbage, labeled as "recyclable" or "garbage". You can use public datasets or create your own by taking pictures of different waste items.
2. **Data Cleaning:** Remove irrelevant images or data points from the dataset. Ensure that the images are of good quality and clearly depict the garbage items. 
3. **Data Augmentation:** Apply data augmentation techniques (e.g., rotation, flipping, scaling) resized (*****)
4. **Data Splitting:** Divide the dataset into training, validation, and testing sets. This allows for proper model evaluation and prevents overfitting.

## Model Selection

- **Convolutional Neural Networks (CNNs):** CNNs are widely used for image classification tasks and have proven to be effective in identifying patterns and features in images. Pre-trained models like ResNet, Inception, or MobileNet can be used as a starting point or fine-tuned for this specific task.
- **Support Vector Machines (SVMs):** SVMs can be used for image classification by extracting features from the images and then training a classifier on those features.
- **Other Models:** Explore other classification models like Random Forests or K-Nearest Neighbors if desired.

## Model Training

1. **Preprocessing:** Resize and normalize the images to a consistent format.
2. **Model Initialization:** Instantiate the chosen model with appropriate hyperparameters.
3. **Training:** Train the model on the training dataset using an optimizer (e.g., Adam, SGD) and a loss function (e.g., categorical cross-entropy). Monitor the training progress and adjust hyperparameters as needed.
4. **Validation:** Evaluate the model's performance on the validation set during training to prevent overfitting and ensure generalization.

## Model Testing

1. **Evaluation:** Evaluate the trained model on the testing dataset to assess its performance on unseen data.
2. **Metrics:** Use appropriate metrics like accuracy, precision, recall, and F1-score to measure the model's performance.
3. **Analysis:** Analyze the results and identify areas for improvement.

## Deployment

1. **Model Saving:** Save the trained model for future use.
2. **Platform:** We chose Gradio as our U/I 
3. **Integration:** Integrate the model into the chosen platform, allowing users to input images or video streams for classification.


