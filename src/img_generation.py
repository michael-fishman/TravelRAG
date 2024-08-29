import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
pipe = pipe.to(device)


def generated_img_prompt(landmark):
    return f"Generate an image of {landmark}, try to make it look realistic, as if its a photo of the place."


def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image


def generate_images(landmarks_list):
    generated_imgs = []
    for landmark in landmarks_list:
        prompt = generated_img_prompt(landmark)
        generated_image = generate_image(prompt)
        generated_imgs.append(generated_image)
    return generated_imgs


def display_img(img):
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # Example prompt
    prompt = "Eiffel Tower in Paris"
    generated_image = generate_image(prompt)
    display_img(generated_image)

    # TODO: change example here
    landmarks_list_example = ["Amsterdam"]
    generated_imgs = generate_images(landmarks_list_example)
    for img in generated_imgs:
        display_img(img)
