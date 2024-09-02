import os.path
from src.prompts import get_travel_plan_prompt, get_prompt_for_creating_full_answer, get_location_recognizer_prompt
import google.generativeai as genai
import json
from PIL import Image

current_dir = os.path.dirname(__file__)
GEMINI_KEY_PATH = os.path.join(current_dir, 'API_keys', 'gemini_api_key.txt')
# with open("./API_keys/gemini_api_key.txt") as f:
with open(GEMINI_KEY_PATH) as f:
    GEMINI_API_KEY = f.read().strip()
genai.configure(api_key=GEMINI_API_KEY)


def get_plan_using_LLM(request: str):
    """
    Generate a travel plan using the Generative AI model.

    Args:
        request (str): The user's request for a travel plan.

    Raises:
        ValueError: The LLM did not return the output in the expected format.

    Returns:
        dict: The generated travel plan.
            travel_plan = {
                'days': days_list,
                'landmarks': landmarks_list,
                'descriptions': descriptions_list
            }
        list: The list of landmarks in the travel plan.
    """

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = get_travel_plan_prompt(request)

    # Generate the travel plan with landmarks
    llm_response = model.generate_content(prompt)
    travel_plan_text = llm_response.text.strip()

    # Initialize empty lists for the days, landmarks, and descriptions
    days_list = []
    landmarks_list = []
    descriptions_list = []

    try:
        # Parse the travel_plan_text as a JSON object
        travel_plan_json = json.loads(travel_plan_text)

        # Ensure the JSON contains an "itinerary" key with a list as its value
        if not isinstance(travel_plan_json, dict) or 'itinerary' not in travel_plan_json:
            raise ValueError("The LLM did not return the output in the expected format.")

        travel_plan_data = travel_plan_json['itinerary']

        if not isinstance(travel_plan_data, list):
            raise ValueError("The itinerary is not in the expected format.")

        # Extract days, landmarks, and descriptions from the travel plan
        for day_plan in travel_plan_data:
            if 'day_of_trip' in day_plan:
                days_list.append(day_plan['day_of_trip'])
            if 'landmark' in day_plan:
                landmarks_list.append(day_plan['landmark'])
            if 'plan_description' in day_plan:
                descriptions_list.append(day_plan['plan_description'])

        # Combine the lists into a dictionary
        travel_plan = {
            'days': days_list,
            'landmarks': landmarks_list,
            'descriptions': descriptions_list
        }

    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError("The LLM did not return the output in the expected format.") from e

    return travel_plan, landmarks_list


def get_landmark_answer_using_LLM(img_query: Image.Image, user_name: str):
    """
    Generate a landmark answer using the Generative AI model

    Args:
        img_query (Image.Image): the image query
        user_name (str): the user's name

    Returns:
        tuple: A tuple containing the full answer and the landmark answer
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Landmark Answer
    prompt = get_location_recognizer_prompt(img_query)
    llm_response = model.generate_content(prompt)
    landmark_answer = llm_response.text.replace('\n', '').strip()
    # Full Answer
    prompt = get_prompt_for_creating_full_answer(user_name, landmark_answer)
    llm_response = model.generate_content(prompt)
    full_answer = llm_response.text
    return full_answer, landmark_answer


def get_landmark_answer_using_RAG(retrieved_answer: str, user_name: str):
    """
    Generate a landmark answer using the Generative AI model.

    Args:
        retrieved_answer (str): The retrieved answer.
        user_name (str): The user's name.

    Returns:
        tuple: A tuple containing the full answer and the retrieved answer.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Full Answer
    prompt = get_prompt_for_creating_full_answer(user_name, retrieved_answer)
    llm_response = model.generate_content(prompt)
    full_answer = llm_response.text
    return full_answer, retrieved_answer


if __name__ == "__main__":
    user_request_1 = "Give me a plan for a trip for 6 days to New York"
    travel_plan, landmarks_list = get_plan_using_LLM(user_request_1)
    print("\nTravel Plan:")
    for day, landmark, description in zip(travel_plan['days'], travel_plan['landmarks'], travel_plan['descriptions']):
        print(f"Day {day}:")
        print(f"  Landmark: {landmark}")
        print(f"  Description: {description}")
        print()
    print("\nLandmarks List:\n", landmarks_list)
