from data import load_user_requests, load_img_text_dataset
from index import init_img_index
from prompts import get_location_recognizer_prompt
from LLM_answers import get_landmark_answer_using_LLM, get_landmark_answer_using_RAG
from retrieve import retrive_landmarks_names
from evaluation import evaluate_landmark_answer, compare_responses
from utils import get_start_time, get_end_time
from time_comparison import save_time_results


# system response pipeline
def get_system_response(img_query, img_index, true_answer=None):
    start_time = get_start_time()
    retrieved_answer = retrive_landmarks_names(img_index, img_query)
    prompt = get_location_recognizer_prompt(request)
    full_answer, landmark_RAG_answer = get_landmark_answer_using_RAG(prompt, retrieved_answer)
    end_time = get_end_time()
    save_time_results(start_time, end_time)
    if true_answer:
        evaluate_landmark_answer(landmark_RAG_answer, true_answer)
    return full_answer


# baseline response pipeline
def get_baseline_response(img_query, true_answer=None):
    start_time = get_start_time()
    prompt = get_location_recognizer_prompt(request)
    full_answer, landmark_LLM_answer = get_landmark_answer_using_LLM(prompt)
    end_time = get_end_time()
    save_time_results(start_time, end_time)
    if true_answer:
        evaluate_landmark_answer(landmark_LLM_answer, true_answer)
    return full_answer

if __name__ == "__main__":
    # User pipeline
    requests = load_user_requests()
    # Prepare Data
    img_text_dataset = load_img_text_dataset()
    # prepare DB
    img_index = init_img_index(img_text_dataset)

    for request in requests:
        system_response = get_system_response(request, img_index)
        baseline_response = get_baseline_response(request)

    compare_responses(system_response, baseline_response)
