import numpy as np
from sentence_transformers import SentenceTransformer
from src.data import load_names, load_images
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_and_embedd_dataset(
        split: str = 'train',
        is_text_index: bool = True,
        embedding_model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
        rec_num: int = 2
) -> tuple:
    """
    Load a dataset and embedd the text field using a sentence-transformer model
    Args:
        dataset_name: The name of the dataset to load
        split: The split of the dataset to load
        embedding_model: The model to use for embedding
        text_field: The field in the dataset that contains the text
        rec_num: The number of records to load and embedd
    Returns:
        tuple: A tuple containing the chunked documents and the embeddings
    """

    print("Loading and embedding the dataset")

    # Load the dataset
    if is_text_index:
        names, formats = load_names()
    else:
        images, names, formats = load_images()
        images = images[:rec_num]
    used_places_names = names[:rec_num]
    used_places_formats = formats[:rec_num]
    print(used_places_names)

    # Embed the first `rec_num` rows of the dataset
    if is_text_index:
        embeddings = get_text_embeddings(used_places_names, model=embedding_model)
    else:
        embeddings = get_img_embeddings(images, model=embedding_model)
    print("Done!")
    return used_places_names, used_places_formats, embeddings


def get_img_embeddings(imgs: list,
    model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")) -> list:
    images_embeddings = []
    for img in imgs:
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        images_embeddings.append(embeddings)

    # Concatenate all embeddings into a single tensor
    embeddings_tensor = torch.cat(images_embeddings, dim=0)

    # Convert the tensor to a NumPy array and then to a list of lists
    embeddings_list = embeddings_tensor.cpu().numpy().tolist()

    return embeddings_list


def get_text_embeddings(
        texts: list,
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
) -> list:
    texts_embeddings = model.encode(texts)

    # Convert the numpy array to a list of lists of floats
    embeddings_as_list = [embedding.tolist() for embedding in texts_embeddings]

    return embeddings_as_list
