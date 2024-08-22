import os
from torchvision import transforms

# Define the path to your dataset
DATASET_PATH = 'datasets/images'
REQUESTS_PATH = "requests.pkl"

# Define a simple image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_names(sample_size=5):
    images_names, images_formats = [], []
    image_paths = [os.path.join(DATASET_PATH, img) for img in os.listdir(DATASET_PATH)]

    # for img_path in image_paths[:sample_size]:
    for img_path in image_paths:
        # extract the image name without the extension
        img_name, img_format = os.path.splitext(os.path.basename(img_path))
        images_names.append(img_name)
        images_formats.append(img_format)

    return images_names, images_formats




def load_user_requests(requests_path):
    # TODO: complete
    raise NotImplementedError


def get_true_images(landmarks_list):
    # TODO: complete
    raise NotImplementedError


