from src.data import load_user_requests, load_img_text_dataset
from src.index import init_img_index
from src.LLM_answers import get_landmark_answer_using_LLM, get_landmark_answer_using_RAG, get_final_landmark_answer_for_user
from src.retrieve import retrive_landmarks_names
from src.evaluation import evaluate_landmark_answer, compare_results_Use_Case_2
from src.utils import get_start_time, get_end_time
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# system response pipeline
def get_RAG_response(img_query, img_index, true_answer=None, id=None, user_name=None):
    start_time = get_start_time()
    retrieved_answer = retrive_landmarks_names(img_index, img_query)
    full_answer, landmark_RAG_answer = get_landmark_answer_using_RAG(retrieved_answer, user_name)
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
def get_baseline_response(img_query, true_answer=None, user_name=None, id=None):
    start_time = get_start_time()
    full_answer, landmark_LLM_answer = get_landmark_answer_using_LLM(img_query, user_name)
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


def inference_pipeline_Use_Case_2(img):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    # Normalize the embeddings (optional but often useful)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    # Prepare Data
    img_text_dataset = load_img_text_dataset()
    # prepare DB
    img_index = init_img_index(img_text_dataset)
    RAG_results = get_RAG_response(image_embeddings, img_index)

if __name__ == "__main__":
    eval_pipeline_Use_Case_2()
