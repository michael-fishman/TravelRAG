import matplotlib.pyplot as plt
from src.embeddings import get_img_embeddings, get_text_embeddings


def retrieve_images(img_index, query_embedding, k):
    # TODO: this code is generated - need to rewrite
    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
    _, indices = img_index.search(query_embedding_np, k)
    return indices


def retrieve_texts(text_index, query_embedding, k):
    # TODO: this code is generated - need to rewrite
    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
    _, indices = text_index.search(query_embedding_np, k)
    return indices


def get_imgs_by_text_indices(text_index, retrieved_indices):
    # TODO: this code is generated - need to rewrite
    return [text_index[idx]['image'].permute(1, 2, 0).numpy() for idx in retrieved_indices[0]]


def get_texts_by_img_indices(img_index, retrieved_indices):
    # TODO: this code is generated - need to rewrite
    return [img_index[idx]['name'] for idx in retrieved_indices[0]]


def retrieve_landmarks_images(text_index, landmark_name_queries):
    query_embeddings = get_text_embeddings(landmark_name_queries)
    retrieved_indices = retrieve_texts(text_index, query_embeddings, k=1)
    retrieved_images = get_imgs_by_text_indices(text_index, retrieved_indices)
    return retrieved_images


def retrieve_landmarks_names(img_index, embedded_img_query):
    retrieved_indices = retrieve_images(img_index, embedded_img_query, k=1)
    retrieved_text_vectors = get_texts_by_img_indices(img_index, retrieved_indices)
    return retrieved_text_vectors


if __name__ == "__main__":
    # TODO: change example here
    landmarks_list_example = ["Amsterdam"]

    retrieved_images = retrieve_landmarks_images(landmarks_list_example)
    # Display the retrieved images
    for retrieved_img in retrieved_images:
        plt.imshow(retrieved_img)
        plt.show()
