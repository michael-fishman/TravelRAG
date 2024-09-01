import os
import pandas as pd
from PIL import Image

# Define the path to your dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = 'src/datasets/images'
TRAVEL_REQUESTS_PATH = 'src/datasets/test_requests_for_UseCase1/travel_requests.csv'
IMAGES_TO_IDENTIFY_PATH = 'src/datasets/test_images_for_UseCase2/images'

def load_names(sample_size=5):
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
    df = pd.read_csv(TRAVEL_REQUESTS_PATH)
    df["id"] = df.index
    return df['id'].to_list(), df['Plan Request'].to_list()
    
def load_user_requests_Use_Case_2():
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