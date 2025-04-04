# downloading a dataset from Kaggle using kagglehub
import pandas as pd
import pickle as pkl
import os
import kagglehub as kh # REQUIRED INSTALLATION
import numpy as np

from PIL import Image

from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

# encoded column names NOTE: if not run, its the default assumption
encoded_y_cols = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

def get_X_y(percent=1, undersample=True, random_state=42, path=None, encoded_y=True, 
            target_size=(150, 150), grayscale=True, minimize_classes=True):
    """function that gets the X and y variables in the proper format given a set of unprocessed images
    Args:
        percent (float): percent of the dataset to use. Default is 1 (100%).
        undersample (bool): whether to undersample the dataset. Default is True.
        random_state (int): random state for reproducibility. Default is 42.
        path (str): path to the downloaded kaggle dataset. Default is None, which will download the dataset.
        encoded_y (bool): whether to one-hot encode the y variable. Default is True.
        target_size (tuple): target size for resizing the images. Default is (150, 150).
        grayscale (bool): whether to convert the images to grayscale. Default is True.
    Returns:
        X (numpy array): processed images.    
        y (pandas dataframe): labels for the images.
    """

    # Download the dataset if not already
    if path is None:
        path = kh.dataset_download("mostafaabla/garbage-classification")

    # create the dataframe
    file_dict = {"file_names": [], "file_paths": [], "classification": []}
    for root, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            file_dict["file_names"].append(file)
            file_dict["file_paths"].append(full_path)
            classification = ''.join([char for char in file if not char.isdigit()])
            classification = classification.split(".")[0]
            file_dict["classification"].append(classification)
    df = pd.DataFrame(file_dict)

    # minimizing the class if necessary
    if minimize_classes:
        # combing glasses
        df.loc[df["classification"].str.contains("glass"), "classification"] = "glass"

        # removing biological
        df = df[df["classification"] != "biological"]

        # combining paper and cardboard
        df.loc[df["classification"].str.contains("paper"), "classification"] = "paper"

        # combinging shoes and clothes
        df.loc[df["classification"].str.contains("shoes"), "classification"] = "clothes"

    # get random sample of size based on percent
    if percent < 1:
        df = df.sample(n=int(len(df) * percent), random_state=random_state)

    # undersample if necessary
    if undersample:
        sampler = RandomUnderSampler(random_state=random_state)
        X = df.drop(columns=["classification"])
        y = df["classification"]
        X, y = sampler.fit_resample(X, y)


    # convert file paths to numpy arrays representing images
    X, errors = get_X(X["file_paths"].values, target_size=target_size, grayscale=grayscale)
    if errors:
        for error in errors:
            print(error)

    # dummy encode y
    if encoded_y:
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.values.reshape(-1, 1))
        columns = [names.split("_")[-1] for names in encoder.get_feature_names_out()]
        global encoded_y_cols
        encoded_y_cols = columns
        y = pd.DataFrame(y, columns=columns)

    return X, y

def get_X(image_paths, target_size=(150, 150), grayscale=True):
    """function that gets the X variable in the proper format given a set of unprocessed images"""
    # processing the images
    processed_imgs = []
    errors = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path)

            # change mode to RGB or L based on grayscale parameter
            if not grayscale and img.mode != 'RGB':
                img = img.convert('RGB')
            elif grayscale and img.mode != 'L':
                img = img.convert('L')
            
            # resize, normalize and append
            img = img.resize(target_size, resample=Image.LANCZOS)
            img = np.array(img).astype(np.float32) / 255.0
            processed_imgs.append(img)
        except Exception as e:
            errors.append(f'FAILED to open image: {image_path}, Error: {e}')

    return np.array(processed_imgs), errors

def get_prediction(model=None, image_path=None, target_size=(150, 150)) -> str:
    """function that gets the predicted class for a given image
    Args:
        model (keras model): trained model. Default is None, which will load the initial model from the pickle file.
        image_path (str): path to the image to be predicted. Default is None.
        target_size (tuple): target size for resizing the images. Default is (150, 150).
    Returns:
        str: predicted class for the image.
    """
    if model is None:
        with open("initial_model.pkl", "rb") as f:
            model = pkl.load(f)

    if model is None:
        raise ValueError("Model must be provided for prediction.")
    if image_path is None:
        raise ValueError("Image path must be provided for prediction.")

    X_input = get_X([image_path], target_size=target_size)
    output = model.predict(X_input)
    predicted_class = np.argmax(output[0])
    global encoded_y_cols

    return encoded_y_cols[predicted_class]