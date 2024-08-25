from src.data import get_true_images
from src.LLM_answers import GEMINI_KEY_PATH
import google.generativeai as genai

with open(GEMINI_KEY_PATH) as f:
    GEMINI_API_KEY = f.read().strip()
genai.configure(api_key=GEMINI_API_KEY)


def evaluate_retrieved_images(retrieved_images, landmarks_list):
    # TODO: complete
    true_images = get_true_images(landmarks_list)
    raise NotImplementedError

def evaluate_generated_images(generated_imgs, landmarks_list):
    # TODO: complete
    true_images = get_true_images(landmarks_list)
    raise NotImplementedError

def evaluate_landmark_answer(landmark_RAG_answer, true_answer):    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    # Construct the prompt to ask the LLM to evaluate the match between the two landmarks
    prompt = (
        f"I have two landmark names. The first is the predicted landmark: '{landmark_RAG_answer}'. "
        f"The second is the true landmark: '{true_answer}'.\n\n"
        "Please evaluate whether the predicted landmark matches the true landmark. "
        "Respond with either 'Correct' or 'Incorrect'."
    )
    
    # Generate the evaluation using the LLM)
    llm_response = model.generate_content(prompt)
    evaluation = llm_response.text.replace("\n", "").strip()
    
    # Ensure the response is either "Correct" or "Incorrect"
    if "Correct" in evaluation:
        return "Correct"
    else:
        return "Incorrect"

def compare_results_Use_Case_1(RAG_results, baseline_results):
    # TODO: complete
    raise NotImplementedError

def compare_results_Use_Case_2(RAG_results, baseline_results):
    # TODO: complete
    raise NotImplementedError