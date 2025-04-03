# downloading a dataset from Kaggle using kagglehub
import pandas as pd
import os
import kagglehub as kh # REQUIRED INSTALLATION
import numpy as np

from PIL import Image
import tempfile
import mmap


from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

def get_X_y(percent=1, undersample=True, random_state=42, path=None, encoded_y=True, target_size=(150, 150)):
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
    X, errors = get_X(X["file_paths"].values, target_size=target_size)
    if errors:
        for error in errors:
            print(error)

    # dummy encode y
    if encoded_y:
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y.values.reshape(-1, 1))
        columns = [names.split("_")[-1] for names in encoder.get_feature_names_out()]
        y = pd.DataFrame(y, columns=columns)

    return X, y

def get_X(image_paths, target_size=(150, 150)):
    """function that gets the X variable in the proper format given a set of unprocessed images"""

    processed_imgs = []
    errors = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(target_size, resample=Image.LANCZOS)
            img = np.array(img).astype(np.float32) / 255.0  # Normalize
            processed_imgs.append(img)
        except Exception as e:
            errors.append(f'FAILED to open image: {image_path}, Error: {e}')

    return np.array(processed_imgs), errors


