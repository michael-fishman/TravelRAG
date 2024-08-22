# from transformers import DALL-E
import matplotlib.pyplot as plt

def generated_img_prompt(landmark):
    # TODO: complete
    return prompt

def generate_image(generator, prompt):
    image = generator(prompt)
    return image

def generate_images(landmarks_list):
    # Assuming the use of a DALL-E model for image generation
    generator = DALL-E.from_pretrained('dalle-mini')

    generated_imgs = []
    for landmark in landmarks_list:
        prompt = generated_img_prompt(landmark)
        generated_image = generate_image(generator, prompt)
        generated_imgs.append(generated_image)
    return generated_imgs

def display_img(img):
    # Display the generated image
    plt.imshow(generated_image)
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
        