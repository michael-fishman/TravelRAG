from datetime import datetime
import urllib.request
import base64
import json
import time
import os
from PIL import Image
from matplotlib import pyplot as plt

webui_server_url = 'http://127.0.0.1:7860'

out_dir = ''
out_dir_t2i = os.path.join(out_dir, 'generated_images')
os.makedirs(out_dir_t2i, exist_ok=True)


def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def encode_file_to_base64(path):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')


def decode_and_save_base64(base64_str, save_path):
    with open(save_path, "wb") as file:
        file.write(base64.b64decode(base64_str))


def call_api(api_endpoint, **payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        f'{webui_server_url}/{api_endpoint}',
        headers={'Content-Type': 'application/json'},
        data=data,
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read().decode('utf-8'))


def call_txt2img_api(landmark, **payload):
    response = call_api('sdapi/v1/txt2img', **payload)
    for index, image in enumerate(response.get('images')):
        save_path = os.path.join(out_dir_t2i, f'{landmark}.png')
        decode_and_save_base64(image, save_path)


def generate_image(landmark):
    payload = {
        "prompt": f"{landmark}, photorealistic, high quality, sharp focus",
        # extra networks also in prompts
        "negative_prompt": "",
        "seed": 1,
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "sampler_name": "Euler a",
        "n_iter": 1,
        "batch_size": 1,

    }
    call_txt2img_api(landmark, **payload)
    # Load and return the generated image
    img_path = os.path.join(out_dir_t2i, f'{landmark}.png')
    img = Image.open(img_path)
    return img


def generate_images(landmarks_list):
    generated_imgs = []
    for landmark in landmarks_list:
        generated_image = generate_image(landmark)
        generated_imgs.append(generated_image)
    return generated_imgs


def display_img(img):
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # Example prompt
    landmark_example = "Vatican City museum, Rome, Italy"
    img = generate_image(landmark_example)
    display_img(img)


