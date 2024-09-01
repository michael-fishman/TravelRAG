import torch
import faiss
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import numpy as np
from src.embeddings import load_and_embedd_dataset
# from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import os

TEXT_INDEX_NAME = "travel-rag-text-index"
IMAGE_INDEX_NAME = "travel-rag-image-index"


current_dir = os.path.dirname(__file__)
PINECONE_KEY_PATH = os.path.join(current_dir, 'API_keys', 'pinecone_api_key.txt')
with open(PINECONE_KEY_PATH) as f:
    PINECONE_API_KEY = f.read().strip()


def init_index(index_name: str, dimension: int, metric: str = 'cosine'):
    """
    Create a pinecone index if it does not exist
    Args:
        index_name: The name of the index
        dimension: The dimension of the index
        metric: The metric to use for the index
    Returns:
        Pinecone: A pinecone object which can later be used for upserting vectors and connecting to VectorDBs
    """
    from pinecone import Pinecone, ServerlessSpec
    print("Creating a Pinecone index...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            # Remember! It is crucial that the metric you will use in your VectorDB will also be a metric your embedding
            # model works well with!
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    print("Done!")
    return pc


def upsert_vectors(
        index: Pinecone,
        embeddings: np.ndarray,
        text_items: list,
        images_formats: list,
        text_field: str = 'Content',
        batch_size: int = 128
)->Pinecone:
    """
    Upsert vectors to a pinecone index
    Args:
        index: The pinecone index object
        embeddings: The embeddings to upsert
        dataset: The dataset containing the metadata
        batch_size: The batch size to use for upserting
    Returns:
        An updated pinecone index
    """
    print("Upserting the embeddings to the Pinecone index...")
    shape = (len(embeddings), len(embeddings[0]))

    ids = [str(i) for i in range(shape[0])]
    meta = [{text_field: text, 'image_format': img_format} for text, img_format in zip(text_items, images_formats)]

    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeddings, meta))

    for i in tqdm(range(0, shape[0], batch_size)):
        # print(to_upsert)
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])
    print("Done!")
    return index


def create_index_and_upsert(is_text_index=True, rec_num=10, embedding_model=SentenceTransformer('all-MiniLM-L6-v2'))->Pinecone:
    """
    Create a pinecone index and upsert the embeddings

    Args:
        is_text_index (bool, optional): . Defaults to True.
        rec_num (int, optional): . Defaults to 10.
        embedding_model (_type_, optional): . Defaults to SentenceTransformer('all-MiniLM-L6-v2').

    Returns:
        Pinecone: A pinecone object which can later be used for upserting vectors and connecting to VectorDBs
    """
    index_name = TEXT_INDEX_NAME if is_text_index else IMAGE_INDEX_NAME
    places_names, images_formats, embeddings = load_and_embedd_dataset(is_text_index=is_text_index, rec_num=rec_num, embedding_model=embedding_model)
    embedding_shape = len(embeddings[0])
    vector_db_index = init_index(index_name, embedding_shape)
    vector_db_index = vector_db_index.Index(index_name)
    index_upserted = upsert_vectors(vector_db_index, embeddings, places_names, images_formats)
    return index_upserted
