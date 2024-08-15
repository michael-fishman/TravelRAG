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
    images_names = []
    image_paths = [os.path.join(DATASET_PATH, img) for img in os.listdir(DATASET_PATH)]

    for img_path in image_paths[:sample_size]:
        # extract the image name without the extension
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        images_names.append(img_name)
    return images_names




def load_user_requests(requests_path):
    # TODO: complete
    raise NotImplementedError


def get_true_images(landmarks_list):
    # TODO: complete
    raise NotImplementedError


