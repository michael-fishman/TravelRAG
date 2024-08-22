from src.data import load_user_requests, load_img_text_dataset
from src.index import init_img_index
from src.prompts import get_location_recognizer_prompt
from src.LLM_answers import get_landmark_answer_using_LLM, get_landmark_answer_using_RAG
from src.retrieve import retrive_landmarks_names
from src.evaluation import evaluate_landmark_answer, compare_results_Use_Case_2
from src.utils import get_start_time, get_end_time


# system response pipeline
def get_RAG_response(img_query, img_index, true_answer=None, id=None):
    start_time = get_start_time()
    retrieved_answer = retrive_landmarks_names(img_index, img_query)
    prompt = get_location_recognizer_prompt(img_query)
    full_answer, landmark_RAG_answer = get_landmark_answer_using_RAG(prompt, retrieved_answer)
    end_time = get_end_time()
    if true_answer:
        correct = evaluate_landmark_answer(landmark_RAG_answer, true_answer)
    # save results
    results = {
        "id": id,
        "full_answer": full_answer,
        "answer": landmark_RAG_answer,
        "retrieved_answer": retrieved_answer,
        "true_answer": true_answer,
        "correct": correct,
        "start_time": start_time,
        "end_time": end_time,
        "response_by": "RAG",
        "use_case": "2"
    }
    return results


# baseline response pipeline
def get_baseline_response(img_query, true_answer=None):
    start_time = get_start_time()
    prompt = get_location_recognizer_prompt(img_query)
    full_answer, landmark_LLM_answer = get_landmark_answer_using_LLM(prompt)
    end_time = get_end_time()
    if true_answer:
        correct = evaluate_landmark_answer(landmark_LLM_answer, true_answer)
    # save results
    results = {
        "id": id,
        "full_answer": full_answer,
        "answer": landmark_LLM_answer,
        "retrieved_answer": None,
        "true_answer": true_answer,
        "correct": correct,
        "start_time": start_time,
        "end_time": end_time,
        "response_by": "Generative Model",
        "use_case": "2"
    }
    return results


def run_full_pipeline_Use_Case_2():
    # User pipeline
    ids, requests, true_answers = load_user_requests()
    # new_requests = load_new_requests() # requests without true answers

    # Prepare Data
    img_text_dataset = load_img_text_dataset()
    # prepare DB
    img_index = init_img_index(img_text_dataset)

    all_RAG_results = []
    all_baseline_results = []
    for id, request, true_answer in zip(ids, requests, true_answers):
        RAG_results = get_RAG_response(request, img_index, true_answer, id)
        baseline_results = get_baseline_response(request, true_answer, id)
        all_RAG_results.append(RAG_results)
        all_baseline_results.append(baseline_results)

    compare_results_Use_Case_2(all_RAG_results, all_baseline_results)


if __name__ == "__main__":
    run_full_pipeline_Use_Case_2()
