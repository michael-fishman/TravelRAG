# Travel RAG
A chatbot that utilizes RAG methodology to retrieve real images of places for travel itineraries
or to identify famous landmarks in images. By using this interface, users can either:
- Enter a text description of their desired trip, or
- Upload an image of a tourist site to identify its location

## Usage example
![image](https://github.com/user-attachments/assets/a53f845e-e4ab-4d9b-b846-63bf0407056b)

## Features
- **Text-based Itinerary Generation**: Users can type in their desired travel destination and duration, and the application will generate a detailed travel plan with suggestions for places to visit alongside their images.
- **Image-based Landmark Identification**: Users can upload an image of a famous tourist landmark. The system identifies the location and generates an appropriate response.
- **Interactive User Interface**: Powered by Streamlit, providing a simple and intuitive interface.

## Installation and Setup
1. **Clone the repository** to your local machine:
   ```bash
   git clone https://github.com/MichaelFish-github/TravelRAG.git
   ```
## Setting up the Environment
1. **Create a Conda environment** using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate the environment**:
   ```bash
   conda activate travelrag
   ```

## Running the Application
1. To launch the Travel Guide UI, run the following command in your terminal:
   ```bash
   streamlit run .\UI\travel_guide_ui.py
   ```
2. Once executed, the Streamlit app will open in your default web browser. 

### Usage
1. **Text-Based Travel Itinerary Generation**:
   - Enter a city name and trip duration (e.g., "3 days in Rome").
   - Click **Generate Itinerary** to get a day-by-day plan of what to do in that city.
   
2. **Image-Based Location Identification**:
   - Upload an image of a landmark using the drag-and-drop functionality or by browsing files.
   - Click **Identify location** to detect the location and receive relevant travel suggestions.
