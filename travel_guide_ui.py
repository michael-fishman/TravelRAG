import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os


# Placeholder function to simulate LLM response and image retrieval
def generate_itinerary(city_name):
    # This is a mock response
    itinerary = {
        "Eiffel Tower": "Day 1: Visit the Eiffel Tower and enjoy the view from the top.",
        "Louvre Museum": "Day 2: Explore the Louvre Museum and see the Mona Lisa.",
        "Notre-Dame Cathedral": "Day 3: Visit the Notre-Dame Cathedral and take a walk along the Seine River."
    }
    return itinerary


def identify_location(image):
    # This is a mock response
    return "Paris", generate_itinerary("Paris")


def get_image(site_name):
    # This is a mock function to return a sample image URL
    images = {
        "Eiffel Tower": os.path.join('selected_images', 'Paris,_Eiffelturm_--_2014_--_1245.jpg'),
        "Louvre Museum": os.path.join('selected_images', '320px-Paris_06_2012_Cour_Napoléon_(Palais_du_Louvre)_Panorama_3004.jpg'),
        "Notre-Dame Cathedral": os.path.join('selected_images', "Paris_75004_Place_de_l'Hôtel-de-Ville_S01_Notre-Dame_remote.jpg")
    }
    return images.get(site_name, None)


# UI Layout
st.title("Travel Guide")
st.write("Tell me where you would like to travel or provide me an image of it.")

user_input = st.text_input("Enter a city name or upload an image",
                           placeholder="Tell me where you would like to travel or provide me an image of it")

uploaded_image = st.file_uploader("Or upload an image of a tourist site", type=["jpg", "jpeg", "png"])

if st.button("Generate Itinerary"):
    if user_input:
        st.subheader(f"Itinerary for {user_input}")
        itinerary = generate_itinerary(user_input)
        for site, description in itinerary.items():
            col1, col2 = st.columns([2, 1])  # 2:1 ratio for text and image columns
            with col1:
                st.write(description)
            with col2:
                image_url = get_image(site)
                if image_url:
                    # response = requests.get(image_url)
                    img = Image.open(image_url)
                    img = img.resize((150, 150))
                    st.image(img, caption=site)

    elif uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        city_name, itinerary = identify_location(image)
        st.subheader(f"Itinerary for {city_name}")
        for site, description in itinerary.items():
            image_url = get_image(site)
            if image_url:
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=site, use_column_width=True)
            st.write(description)
    else:
        st.error("Please enter a city name or upload an image.")

