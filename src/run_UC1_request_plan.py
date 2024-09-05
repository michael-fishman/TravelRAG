from src.data import load_user_requests_Use_Case_1
from src.index import create_index_and_upsert
from src.LLM_answers import get_plan_using_LLM
from src.retrieve import retrieve_landmarks_images
from src.img_generation import generate_images
from src.evaluation import evaluate_retrieved_images, evaluate_generated_images, compare_results_Use_Case_1, \
    save_results_Use_Case_1
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime


# system response pipeline
def get_RAG_response(request: str, text_index, id=None, eval=False) -> dict:
    """
    Get response from RAG model.

    Args:
        request (str): the user's request for a travel plan.
        text_index (_type_): Pincone index for text embeddings.
        id (_type_, optional): the request id. Defaults to None.
        eval (bool, optional): wether to eval or not . Defaults to False.

    Returns:
        dict: the response from the RAG model.
    """
    start_time = datetime.now()
    travel_plan, landmarks_list = get_plan_using_LLM(request)
    retrieved_images, retrieved_names = retrieve_landmarks_images(text_index, landmarks_list, return_names=True)
    end_time = datetime.now()
    if eval:
        accuracy, evaluation = evaluate_retrieved_images(retrieved_names, landmarks_list)
    else:
        accuracy, evaluation = None
    # save results
    results = {
        "id": id,
        "travel_plan": travel_plan,
        "landmarks_list": landmarks_list,
        "images": retrieved_images,
        "accuracy": accuracy,
        "evaluation": evaluation,
        "start_time": start_time,
        "end_time": end_time,
        "response_by": "RAG",
        "use_case": "1"
    }
    return results


# baseline response pipeline
def get_baseline_response(request: str, id: int = None, eval: bool = False) -> dict:
    """
    Get response from the baseline model.

    Args:
        request (str): the user's request for a travel plan.
        id (int, optional): the request id. Defaults to None.
        eval (bool, optional): wether to eval or not . Defaults to False.

    Returns:
        dict: the response from the baseline model.
    """
    start_time = datetime.now()
    travel_plan, landmarks_list = get_plan_using_LLM(request)
    generated_imgs = generate_images(landmarks_list)
    end_time = datetime.now()
    if eval:
        accuracy, evaluation = evaluate_generated_images(generated_imgs, landmarks_list)
    else:
        accuracy = None
    # save results
    results = {
        "id": id,
        "travel_plan": travel_plan,
        "landmarks_list": landmarks_list,
        "images": generated_imgs,
        "accuracy": accuracy,
        "evaluation": evaluation,
        "start_time": start_time,
        "end_time": end_time,
        "response_by": "Generative Model",
        "use_case": "1"
    }
    return results


def eval_pipeline_Use_Case_1():
    # User pipeline
    ids, requests = load_user_requests_Use_Case_1()
    # Prepare Data
    text_index = create_index_and_upsert(rec_num=-1)

    for id, request in zip(ids, requests):
        RAG_results = get_RAG_response(request, text_index, id, eval=True)
        baseline_results = get_baseline_response(request, id, eval=True)
        save_results_Use_Case_1(RAG_results, baseline_results)



def inference_pipeline_Use_Case_1(query):
    # Prepare DB
    text_index = create_index_and_upsert(rec_num=50, embedding_model=SentenceTransformer('all-MiniLM-L6-v2'))
    # Get RAG response
    RAG_results = get_RAG_response(query, text_index)
    travel_plan = RAG_results["travel_plan"]
    images = RAG_results['images']
    return travel_plan, images


def test_pipeline():
    # Query Example
    query = "Plan a 2 week trip to Italy"
    # Get Results
    travel_plan, images_list = inference_pipeline_Use_Case_1(query)
    print(travel_plan)


if __name__ == "__main__":
    eval_pipeline_Use_Case_1()
