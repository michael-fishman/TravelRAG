from src.data import get_true_images
from transformers import pipeline

def evaluate_retrieved_images(retrieved_images, landmarks_list):
    # TODO: complete
    true_images = get_true_images(landmarks_list)
    raise NotImplementedError

def evaluate_generated_images(generated_imgs, landmarks_list):
    # TODO: complete
    true_images = get_true_images(landmarks_list)
    raise NotImplementedError

def evaluate_retrieved_images(landmark_RAG_answer, true_answer):
    # TODO: complete
    raise NotImplementedError

def evaluate_landmark_answer(landmark_RAG_answer, true_answer):
    llm = pipeline("text-generation", model="gpt-3.5-turbo")
    
    # Construct the prompt to ask the LLM to evaluate the match between the two landmarks
    prompt = (
        f"I have two landmark names. The first is the predicted landmark: '{landmark_RAG_answer}'. "
        f"The second is the true landmark: '{true_answer}'.\n\n"
        "Please evaluate whether the predicted landmark matches the true landmark. "
        "Respond with either 'Correct' or 'Incorrect'."
    )
    
    # Generate the evaluation using the LLM
    llm_response = llm(prompt, max_length=50, num_return_sequences=1)
    evaluation = llm_response[0]['generated_text'].strip()
    
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