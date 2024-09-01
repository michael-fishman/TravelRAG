import matplotlib.pyplot as plt
from PIL import Image
from src.embeddings import get_text_embeddings
from pinecone import QueryResponse
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(current_dir, '../datasets/images/')


def retrieve_neighbors(upserted_index, query_embedding, k=5):
    query_result: QueryResponse = upserted_index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    # Print the query results
    print("Query results:")
    for match in query_result.matches:
        print(f"ID: {match.id}, Score: {match.score}, Metadata: {match.metadata}")
    return [match for match in query_result.matches]


def get_imgs_by_text_indices(matches):
    images = []
    for match in matches:
        img_path = DATASET_PATH + match.metadata.get('Content') + match.metadata.get('image_format')
        if img_path:
            img = Image.open(img_path)
            images.append(img)
    return images


def get_texts_by_img_indices(matches):
    texts = []
    for match in matches:
        text = match.metadata.get('Content')
        if text:
            texts.append(text)

    return texts


def retrieve_landmarks_images(text_index, landmark_name_queries, return_names=False):
    retrieved_images, images_names = [], []
    print(f'landmark_name_queries = {landmark_name_queries}')
    query_embeddings = get_text_embeddings(landmark_name_queries)
    for query_embedding in query_embeddings:
        matches = retrieve_neighbors(text_index, query_embedding, k=1)
        images_names.extend([match.metadata.get('Content') for match in matches])
        retrieved_images.extend(get_imgs_by_text_indices(matches))
    if return_names:
        return retrieved_images, images_names
    else:
        return retrieved_images


def retrieve_landmarks_names(img_index, embedded_img_query):
    matches = retrieve_neighbors(img_index, embedded_img_query, k=1)
    retrieved_text_vectors = get_texts_by_img_indices(matches)
    return retrieved_text_vectors


if __name__ == "__main__":
    # TODO: change example here
    landmarks_list_example = ["Amsterdam"]

    retrieved_images = retrieve_landmarks_images(landmarks_list_example)
    # Display the retrieved images
    for retrieved_img in retrieved_images:
        plt.imshow(retrieved_img)
        plt.show()
