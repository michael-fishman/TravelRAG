import numpy as np
from sentence_transformers import SentenceTransformer
from data import load_names


def load_and_embedd_dataset(
        split: str = 'train',
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
        rec_num: int = 2
) -> tuple:
    """
    Load a dataset and embedd the text field using a sentence-transformer model
    Args:
        dataset_name: The name of the dataset to load
        split: The split of the dataset to load
        model: The model to use for embedding
        text_field: The field in the dataset that contains the text
        rec_num: The number of records to load and embedd
    Returns:
        tuple: A tuple containing the chunked documents and the embeddings
    """

    print("Loading and embedding the dataset")

    # Load the dataset
    names, formats = load_names()

    # Chunk the documents
    used_places_names = names[:rec_num]
    used_places_formats = formats[:rec_num]
    print(used_places_names)

    # Embed the first `rec_num` rows of the dataset
    embeddings = model.encode(used_places_names)

    print("Done!")
    return used_places_names, used_places_formats, embeddings


def get_img_query_embeddings(img_queries):
    # TODO: complete
    # (model, img_query.unsqueeze(0))
    # return query_embedding
    raise NotImplementedError


def get_text_query_embeddings(
        text_query: str,
        model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2'),
) -> np.array:
    # query_embedding = model.encode(text_queries)
    query_embedding = [float(val) for val in list(model.encode(text_query))]
    return query_embedding
