import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

# Define the path to your dataset
DATASET_PATH = 'landmark_dataset.pkl'
REQUESTS_PATH = "requests.pkl"

# Define a simple image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_img_text_dataset(sample_size=500):
    images = []
    image_paths = [os.path.join(DATASET_PATH, img) for img in os.listdir(DATASET_PATH)]

    for img_path in image_paths[:sample_size]:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)
    torch.stack(images)

    # TODO: need to change this function so we will build the img - text dataset
    return img_text_dataset


def load_user_requests(requests_path):
    # TODO: complete
    raise NotImplementedError


def get_true_images(landmarks_list):
    # TODO: complete
    raise NotImplementedError

if __name__ == "__main__":
    # Load the images
    img_text_dataset = load_img_text_dataset()
    # Load the requests
    requests = load_user_requests()
