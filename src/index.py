import torch
import faiss
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import numpy as np
from torchvision.models import resnet50
from src.embeddings import load_and_embedd_dataset

TEXT_INDEX_NAME = "TravelRAG"

with open("./API_keys/pinecone_api_key.txt") as f:
    PINECONE_API_KEY = f.read().strip()


# Create embeddings
def create_embeddings(model, images):
    with torch.no_grad():
        embeddings = model(images).squeeze()
    return embeddings


def init_img_index(images):
    # Load a pre-trained ResNet model
    model = resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer

    # Generate embeddings for the images
    embeddings = create_embeddings(model, images)

    # Convert embeddings to numpy for Faiss
    embeddings_np = embeddings.cpu().numpy()

    # Initialize a Faiss index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index


def init_text_index(index_name: str, dimension: int, metric: str = 'cosine'):
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
        chunked_docs: list,
        text_field: str = 'Content',
        batch_size: int = 128
):
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
    shape = embeddings.shape

    ids = [str(i) for i in range(shape[0])]
    meta = [{text_field: text} for text in chunked_docs]

    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeddings, meta))

    for i in tqdm(range(0, shape[0], batch_size)):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])
    return index


def create_index_and_upsert():
    places_names, embeddings = load_and_embedd_dataset()
    embedding_shape = embeddings.shape[1]
    text_index = init_text_index(TEXT_INDEX_NAME, embedding_shape)
    index = text_index.Index(TEXT_INDEX_NAME)
    index_upserted = upsert_vectors(index, embeddings, places_names)
    return index_upserted


