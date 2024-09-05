import os
import pandas as pd
from PIL import Image

# Define the path to your dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = './datasets/images'
TRAVEL_REQUESTS_PATH = './datasets/test_requests_for_UseCase1/test.csv'
IMAGES_TO_IDENTIFY_PATH = './datasets/test_images_for_UseCase2/images'


def load_names(sample_size: int = 5):
    """
    Load the names of the images in the dataset

    Args:
        sample_size (int, optional): . Defaults to 5.

    Returns:
        list: List of image names
        list: List of image formats
    """
    images_names, images_formats = [], []
    print(f"Data set path is: {DATASET_PATH}")
    image_paths = [os.path.join(DATASET_PATH, img) for img in os.listdir(DATASET_PATH)]

    # for img_path in image_paths[:sample_size]:
    for img_path in image_paths:
        # extract the image name without the extension
        img_name, img_format = os.path.splitext(os.path.basename(img_path))
        if img_format == ".avif":
            print(img_name)
        images_names.append(img_name)
        images_formats.append(img_format)

    return images_names, images_formats


def load_images():
    """
    Load the images in the dataset

    Returns:
        list: List of images
        list: List of image names
        list: List of image formats
    """
    images, images_names, images_formats = [], [], []
    image_paths = [os.path.join(DATASET_PATH, img) for img in os.listdir(DATASET_PATH)]

    # for img_path in image_paths[:sample_size]:
    for img_path in image_paths:
        img = Image.open(img_path)
        img_name, img_format = os.path.splitext(os.path.basename(img_path))
        images.append(img)
        images_names.append(img_name)
        images_formats.append(img_format)
    return images, images_names, images_formats


def load_user_requests_Use_Case_1():
    """
    Load the user requests for Use Case 1

    Returns:
        list: List of user request ids
        list: List of user requests
    """
    df = pd.read_csv(TRAVEL_REQUESTS_PATH)
    df["id"] = df.index
    return df['id'].to_list(), df['Plan Request'].to_list()


def load_user_requests_Use_Case_2():
    """
    Load the user requests for Use Case 2

    Returns:
        list: List of user request ids
        list: List of user requests
        list: List of true answers
    """
    images, images_names, images_formats, ids = [], [], [], []
    image_paths = [os.path.join(IMAGES_TO_IDENTIFY_PATH, img) for img in os.listdir(IMAGES_TO_IDENTIFY_PATH)]

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        img_name, img_format = os.path.splitext(os.path.basename(img_path))
        images.append(img)
        images_names.append(img_name)
        images_formats.append(img_format)
        ids.append(i)

    return ids, images, images_names


if __name__ == "__main__":
    # load_user_requests_Use_Case_1()
    # load_user_requests_Use_Case_2()
    load_names()
