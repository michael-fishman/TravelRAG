import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os
from src.embeddings import get_text_embeddings
from src.retrieve import retrieve_landmarks_images
from src.index import create_index_and_upsert
from src.run_UC1_request_plan import get_RAG_response as get_RAG_response_UC1
from src.run_UC2_request_to_location import get_RAG_response as get_RAG_response_UC2
from src.data import DATASET_PATH


# Placeholder function to simulate LLM response and image retrieval
# def generate_itinerary(user_request: str, text_index):
    # # This is a mock response
    # itinerary = {
    #     "Eiffel Tower": "Day 1: Visit the Eiffel Tower and enjoy the view from the top.",
    #     "Louvre Museum": "Day 2: Explore the Louvre Museum and see the Mona Lisa.",
    #     "Notre-Dame Cathedral": "Day 3: Visit the Notre-Dame Cathedral and take a walk along the Seine River."
    # }
    # rag_response_dict = get_RAG_response(request=user_request, text_index=text_index)
    # return rag_response_dict.travel_plan


# def identify_location(image):
#     # This is a mock response
#     return "Paris", generate_itinerary("Paris")


# def get_image(site_name: str, text_index_upserted):
    # query_embedding = get_text_embeddings(texts=[site_name])
    # query_result = index_upserted.query(vector=query_embedding, top_k=1,include_metadata=True)
    # image = retrieve_landmarks_images(text_index=text_index_upserted, landmark_name_queries=site_name)
    # img_name = query_result.matches[0].metadata['Content']
    # img_format = query_result.matches[0].metadata['image_format']
    # retrieved_image_url = os.path.join(DATASET_PATH, f'{img_name}{img_format}')
    # print(f'site_name = {site_name}, retrieved_image_url = {retrieved_image_url}')
    # print(f'site_name = {site_name}, image = {image.filename}')
    # return image

    # # This is a mock function to return a sample image URL
    # images = {
    #     "Eiffel Tower": os.path.join('TravelRAG/datasets/images', 'Eiffel_Tower.jpg'),
    #     "Louvre Museum": os.path.join('TravelRAG/datasets/images', '320px-Paris_06_2012_Cour_Napoléon_(Palais_du_Louvre)_Panorama_3004.jpg'),
    #     "Notre-Dame Cathedral": os.path.join('TravelRAG/datasets/images', "Paris_75004_Place_de_l'Hôtel-de-Ville_S01_Notre-Dame_remote.jpg")
    # }
    # return images.get(site_name, None)


def resize_image_to_max(img, max_size=200):
    """Resize image to maintain aspect ratio, with max width/height of max_size."""
    width, height = img.size
    if width > height:
        ratio = max_size / float(width)
        new_size = (max_size, int(height * ratio))
    else:
        ratio = max_size / float(height)
        new_size = (int(width * ratio), max_size)
    return img.resize(new_size)

# Change image names
# for img_pth, new_pth in zip(image_paths, new_paths):
#     new_pth = f"{new_pth[:new_pth.rfind('.')]}{img_pth[img_pth.rfind('.'):]}"
#     print(f'img_pth = {img_pth}, new_pth = {new_pth}')
#     os.rename(img_path, new_pth)


text_index_upserted = None
img_index_upserted = None

# UI Layout
st.title("Travel Guide")
st.write("Tell me where you would like to travel or provide me an image of it.")

# user_input = st.text_input("Enter a city name or upload an image",
#                            placeholder="Tell me where you would like to travel or provide me an image of it")
#
# uploaded_image = st.file_uploader("Or upload an image of a tourist site", type=["jpg", "jpeg", "png"])

with st.form(key='travel_form'):
    user_input = st.text_input("Enter a city name or upload an image", placeholder="Tell me where you would like to travel or provide me an image of it")
    uploaded_image = st.file_uploader("Or upload an image of a tourist site", type=["jpg", "jpeg", "png"])
    submit_button = st.form_submit_button(label='Generate Itinerary')

# if st.button("Generate Itinerary"):
if submit_button:
    if text_index_upserted is None:
        # Initialize and upsert data to the index
        text_index_upserted = create_index_and_upsert(rec_num=370, is_text_index=True)
    if user_input:
        st.subheader(f"Here is the itinerary:")
        results_dict = get_RAG_response_UC1(request=user_input, text_index=text_index_upserted)
        travel_plan_dict, retreived_images = results_dict.get('travel_plan'), results_dict.get('images')
        days, landmarks, descriptions = travel_plan_dict.get('days'), travel_plan_dict.get('landmarks'), travel_plan_dict.get('descriptions')
        for day, landmark, description, img in zip(days, landmarks, descriptions, retreived_images):
            col1, col2 = st.columns([2, 1])  # 2:1 ratio for text and image columns
            with col1:
                st.write(f'Day {day}: {description}')
            with col2:
                print(f'landmark = {landmark}, img.filename = {img.filename}')
                img = resize_image_to_max(img, 200)
                st.image(img, caption=landmark)

    elif uploaded_image:
        if img_index_upserted is None:
            # Initialize and upsert data to the index
            img_index_upserted = create_index_and_upsert(rec_num=370, is_text_index=False)
        image = Image.open(uploaded_image)
        # st.image(image, caption="Uploaded Image", use_column_width=True)
        # city_name, itinerary = identify_location(image)
        st.subheader(f"Here is the itinerary:")
        results_dict = get_RAG_response_UC2(img_query=image, img_index=text_index_upserted)
        travel_plan_dict, retreived_images = results_dict.get('travel_plan'), results_dict.get('images')
        days, landmarks, descriptions = travel_plan_dict.get('days'), travel_plan_dict.get(
            'landmarks'), travel_plan_dict.get('descriptions')
        for day, landmark, description, img in zip(days, landmarks, descriptions, retreived_images):
            col1, col2 = st.columns([2, 1])  # 2:1 ratio for text and image columns
            with col1:
                st.write(f'Day {day}: {description}')
            with col2:
                print(f'landmark = {landmark}, img.filename = {img.filename}')
                img = resize_image_to_max(img, 200)
                st.image(img, caption=landmark)
    else:
        st.error("Please enter a city name or upload an image.")


