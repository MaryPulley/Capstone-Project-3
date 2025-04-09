# Project 3-Garbage Classification Model

## Team 6
Asa Adomatis, Mary Pulley, Vikram Borra, Sophak So, Katie Craig
TA- Rimi Sharma

![image](https://github.com/user-attachments/assets/ab6769bd-7fae-4531-800b-ddf63dbe967e)


## Overview

This project aims to develop an image classification model using a convolutional neural network (CNN) architecture to determine whether an object is recyclable. The model classifies images into one of 8 categories, each corresponding to a type of recyclable material. By leveraging machine learning and computer vision, this project provides users with a quick decision-making tool to promote recycling.


## Instructions to Run Code
To use User Interface Application - "Waste Classifier App" Download the following files to utilize with the UI_code_final.ipynb notebook

garbage_recycle_model.pkl

preprocessing.py

Folder EXAMPLE IMAGES

## Package Install

1. pip install tensorflow 
2. pip install pandas
3. pip install Pillow
4. pip install scikit-learn
5. pip install kagglehub
6. pip install imblearn (imbalanced-learn)
7. pip install gdown

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

![image](https://github.com/user-attachments/assets/c611349b-0058-4803-8da0-32ea5bcf877b)



## Model Training

1. **Preprocessing:** Resize and normalize the images to a consistent format.
2. **Model Initialization:** Instantiate the chosen model with appropriate hyperparameters.
3. **Training:** The model was trained for 10 epochs with a batch size of 32 using an optimizer and categorical cross-entropy loss. Training accuracy remained high throughout, ranging from 97.2% to 98.7%.
4. **Validation:** The model’s validation accuracy peaked at 89.8% in the final epoch, indicating strong generalization and effective overfitting control.

![image](https://github.com/user-attachments/assets/55acf8e1-13bc-42d1-91a5-509c5efb3861)


## Model Testing

1.⏱️ Test time: Approximately 13 seconds, with an average of 227ms per step.

2.📉 Test loss: 0.6613 — indicating relatively low error on unseen data.

3.✅ Test accuracy: 88.03%, showing strong performance on generalizing to new samples.

4.📊 Compiled metric (accuracy): 89.79%, consistent with the final validation accuracy during training.


![image](https://github.com/user-attachments/assets/5daad6e9-e863-45ab-8888-9f354e105578)

![image](https://github.com/user-attachments/assets/d8611443-1387-4ff6-9e37-e5cf29731ae3)


## Usage

1. Download the dataset from the link above.
2. Install the required packages.
3. Run the code to train the model.
4. Use the link to gradio (UI_code_final.ipynb) 

## Deployment

1. **Model Saving:** Save the trained model for future use.
2. **Platform:** We chose Gradio as our U/I 
3. **Integration:** Integrate the model into the chosen platform, allowing users to input images or video streams for classification.


