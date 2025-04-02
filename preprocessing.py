# downloading a dataset from Kaggle using kagglehub
import pandas as pd
import os
import kagglehub as kh # REQUIRED INSTALLATION
import numpy as np

from PIL import Image
import tempfile
import mmap
import pickle

def _import_images(image_paths) -> list[Image]:
    images = []
    errors = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path))
        except Exception as e:
            errors.append(f'FAILED to open image: {image_path}, Error: {e}')

    return images, errors

def get_images_and_df(path=None):
    """function that gets the X variable in the proper format from the datset"""
    if path is None:
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

def get_X(images, verbose=False, target_size=None):
    """function that gets the X variable in the proper format given a set of unprocessed images"""

    if target_size is None:
        sizes = set([img.size for img in images])
        x_size = [shape[0] for shape in sizes]
        y_size = [shape[1] for shape in sizes]

        # get the median x and y size
        x_size = int(np.median(x_size))
        y_size = int(np.median(y_size))
        target_size = (x_size, y_size)

    # Create a temporary file to store the processed images
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    # Calculate the size of each image array
    processed_imgs_shape = (len(images), target_size[1], target_size[0], 3)  # Assuming RGB images
    img_size = target_size[0] * target_size[1] * 3 * np.float32().itemsize
    
    # Memory-map the file for storing the processed images
    with open(temp_file.name, 'wb') as f:
        f.truncate(len(images) * img_size)

    processed_imgs = np.memmap(temp_file.name, dtype=np.float32, mode='r+', shape=processed_imgs_shape)

    for i, img in enumerate(images):
        # Convert grayscale images to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            if verbose:
                print(f"Image {i} required to be converted to RGB")
        processed_img = img.resize(target_size, resample=Image.LANCZOS)
        processed_img = np.array(processed_img).astype(np.float32)
        print("processed img shape", processed_img.shape)
        processed_img = processed_img / 255.0
        print("processed img shape post refinement", processed_img.shape)
        processed_imgs[i] = processed_img

    # Delete the temporary file
    os.remove(temp_file.name)

    return processed_imgs

