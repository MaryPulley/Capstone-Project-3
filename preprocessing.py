# downloading a dataset from Kaggle using kagglehub
import pandas as pd
import os
import kagglehub as kh # REQUIRED INSTALLATION
import numpy as np

from PIL import Image
import tempfile
import mmap

def _import_images(image_paths) -> list[Image]:
    images = []
    errors = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path))
        except Exception as e:
            errors.append(f'FAILED to open image: {image_path}, Error: {e}')

    return images, errors

def get_images_and_df():
    """function that gets the X variable in the proper format from the datset"""

    path = kh.dataset_download("mostafaabla/garbage-classification")
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

    images, errors = _import_images(df["file_paths"])



    return images, df

def get_X(images, verbose=False):
    """function that gets the X variable in the proper format given a set of unprocessed images"""

    
    sizes = set([img.size for img in images])
    x_size = [shape[0] for shape in sizes]
    y_size = [shape[1] for shape in sizes]

    # get the median x and y size
    x_size = int(np.median(x_size))
    y_size = int(np.median(y_size))
    target_size = (x_size, y_size)

    processed_imgs = []
    for img in images:
        processed_img = img.resize(target_size, resample=Image.LANCZOS)
        processed_img = np.array(processed_img).astype(np.float32) / 255.0  # Normalize
        processed_imgs.append(processed_img)

    return processed_imgs

