from src.data import load_user_requests, load_images
from src.embeddings import get_img_embeddings
from src.index import init_img_index, create_index_and_upsert
from src.prompts import get_location_recognizer_prompt
from src.LLM_answers import get_landmark_answer_using_LLM, get_landmark_answer_using_RAG
from src.retrieve import retrieve_landmarks_names
from src.evaluation import evaluate_landmark_answer, compare_results_Use_Case_2
from src.utils import get_start_time, get_end_time
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# system response pipeline
def get_RAG_response(embedded_query, img_index, true_answer=None, id=None):
    start_time = get_start_time()
    retrieved_answer = retrieve_landmarks_names(img_index, embedded_query)
    prompt = get_location_recognizer_prompt(embedded_query)
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


def eval_pipeline_Use_Case_2():
    # User pipeline
    ids, requests, true_answers = load_user_requests()
    # new_requests = load_new_requests() # requests without true answers

    # Prepare Data
    images = load_images()
    # prepare DB
    img_index = init_img_index(images)

    all_RAG_results = []
    all_baseline_results = []
    for id, request, true_answer in zip(ids, requests, true_answers):
        RAG_results = get_RAG_response(request, img_index, true_answer, id)
        baseline_results = get_baseline_response(request, true_answer, id)
        all_RAG_results.append(RAG_results)
        all_baseline_results.append(baseline_results)

    compare_results_Use_Case_2(all_RAG_results, all_baseline_results)


def inference_pipenine_Use_Case_2(img):
    # prepare DB
    img_index = create_index_and_upsert(is_text_index=False, rec_num=50)

    # Embedding
    query_embedding = get_img_embeddings(img)

    RAG_results = get_RAG_response(query_embedding, img_index)
    full_answer = RAG_results["full_answer"]
    retrieved_answer = RAG_results["retrieved_answer"]
    return full_answer, retrieved_answer


if __name__ == "__main__":
    eval_pipeline_Use_Case_2()
