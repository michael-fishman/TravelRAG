from src.prompts import get_travel_plan_prompt, get_prompt_for_creating_full_answer
import google.generativeai as genai
from src.prompts import get_location_recognizer_prompt
import json

with open("./API_keys/gemini_api_key.txt") as f:
    GEMINI_API_KEY = f.read().strip()
genai.configure(api_key=GEMINI_API_KEY)

def get_plan_using_LLM(request):
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

def get_landmark_answer_using_LLM(img_query, user_name):
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

def get_landmark_answer_using_RAG(retrieved_answer, user_name):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Full Answer
    prompt = get_prompt_for_creating_full_answer(user_name, retrieved_answer)
    llm_response = model.generate_content(prompt)
    full_answer = llm_response.text
    return full_answer, retrieved_answer
    

if __name__ == "__main__":
    user_request_1 = "Give me a plan for a trip for 2 weeks to Amsterdam"
    prompt_example = get_travel_plan_prompt(user_request_1)
    travel_plan, landmarks_list = get_plan_using_LLM(prompt_example)
    print("Travel Plan:\n", travel_plan)
    print("\nLandmarks List:\n", landmarks_list)