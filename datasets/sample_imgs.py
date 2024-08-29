import os
import csv
import google.generativeai as genai
import pandas as pd


# Specify the directory path
directory_path = 'datasets/images'

# Count the number of files
file_count = len([file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))])

print(f"Number of files in '{directory_path}': {file_count}")

#  Get a list of all files in the directory
files = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
# Specify the output CSV file
output_csv = 'datasets/images_names.csv'
# Write the file names to a CSV file
with open(output_csv, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header
    writer.writerow(['File Name'])
    # Write the file names
    for file in files:
        writer.writerow([file])

print(f"File names have been written to '{output_csv}'.")

places_df = pd.read_csv('datasets/images_names.csv')

def add_country_columns(places_df):
    current_dir = os.path.dirname(__file__)
    GEMINI_KEY_PATH = os.path.join(os.path.abspath(os.path.join(current_dir, os.pardir)), 'src/API_keys', 'gemini_api_key.txt')
    # with open("./API_keys/gemini_api_key.txt") as f:
    with open(GEMINI_KEY_PATH) as f:
        GEMINI_API_KEY = f.read().strip()
    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    countries = []
    for location in places_df['File Name']:
        prompt = f"name the country of this place: {location}. \nreturn only country name without additional chars."
        # Generate the travel plan with landmarks
        country = model.generate_content(prompt).text.strip()
        countries.append(country)
        
    places_df["Country"] = countries
    places_df.to_csv('datasets/images_names_countries.csv', index=False)
