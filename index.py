import torch
import faiss
from torchvision.models import resnet50
from data import load_img_text_dataset


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


def init_text_index(img_text_dataset):
    # TODO: complete
    raise NotImplementedError


if __name__ == "__main__":
    # Prepare Data
    img_text_dataset = load_img_text_dataset()
    img_index = init_img_index(img_text_dataset)
    text_index = init_text_index(img_text_dataset)
