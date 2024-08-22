from transformers import pipeline
from src.prompts import get_travel_plan_prompt
import google.generativeai as genai
from src.prompts import get_location_recognizer_prompt

with open("./API_keys/gemini_api_key.txt") as f:
    GEMINI_API_KEY = f.read().strip()
genai.configure(api_key=GEMINI_API_KEY)

def get_plan_using_LLM(prompt):
    # TODO: this code is generated - need to rewrite
    # Load the LLM (e.g., using OpenAI GPT or similar model)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Generate the travel plan with landmarks
    llm_response = model.generate_content(prompt)
    travel_plan = llm_response.text

    # Extract the travel plan and landmarks list
    try:
        # Split the output into the travel plan and landmarks section
        travel_plan_section, landmarks_section = travel_plan.split("Landmarks:")
        
        # Clean up the travel plan section
        travel_plan_section = travel_plan_section.strip()

        # Extract the list of landmarks
        landmarks_list = [landmark.strip() for landmark in landmarks_section.split(",")]

        # Output the results
        print("Travel Plan:\n", travel_plan_section)
        print("Landmarks List:\n", landmarks_list)

    except ValueError:
        raise ValueError("The LLM did not return the output in the expected format.")

    return travel_plan, landmarks_list

def get_landmark_answer_using_LLM(img_query):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = get_location_recognizer_prompt(img_query)
    llm_response = model.generate_content(prompt)
    landmark_answer = llm_response.text.replace('\n', '').strip()
    return landmark_answer

def get_landmark_answer_using_RAG(img_query, retrieved_answer):
    # TODO: this code is generated - need to rewrite
    # TODO: complete, somehow change that the LLM will use the retrieved_answer if not None (this is the RAG part)
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    prompt = get_location_recognizer_prompt(img_query)
    llm_response = model.generate_content(prompt)
    landmark_answer = llm_response.text
    return landmark_answer

def create_final_travel_plan(travel_plan, retrieved_images):
    # TODO: complete
    raise NotImplementedError

if __name__ == "__main__":
    user_request_1 = "Give me a plan for a trip for 2 weeks to Amsterdam"
    prompt_example = get_travel_plan_prompt(user_request_1)
    travel_plan, landmarks_list = get_plan_using_LLM(prompt_example)
    print("Travel Plan:\n", travel_plan)
    print("\nLandmarks List:\n", landmarks_list)