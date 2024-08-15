from transformers import pipeline
from prompts import get_travel_plan_prompt

def get_plan_using_LLM(prompt):
    # Load the LLM (e.g., using OpenAI GPT or similar model)
    llm = pipeline("text-generation", model="gpt-3.5-turbo")

    # Generate the travel plan with landmarks
    llm_response = llm(prompt, max_length=700, num_return_sequences=1)
    travel_plan = llm_response[0]['generated_text']

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

def get_landmark_answer_using_LLM(prompt):
    # Load the LLM (e.g., using OpenAI GPT or similar model)
    llm = pipeline("text-generation", model="gpt-3.5-turbo")

    # Generate the travel plan with landmarks
    llm_response = llm(prompt, max_length=700, num_return_sequences=1)
    landmark_answer = llm_response[0]['generated_text']
    return landmark_answer

def get_landmark_answer_using_RAG(prompt, retrieved_answer):
    # TODO: complete, somehow change that the LLM will use the retrieved_answer if not None (this is the RAG part)
    
    # Load the LLM (e.g., using OpenAI GPT or similar model)
    llm = pipeline("text-generation", model="gpt-3.5-turbo")

    # Generate the travel plan with landmarks
    llm_response = llm(prompt, max_length=700, num_return_sequences=1)
    landmark_answer = llm_response[0]['generated_text']
    raise NotImplementedError
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