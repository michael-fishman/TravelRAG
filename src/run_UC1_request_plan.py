from src.data import load_user_requests
from src.index import create_index_and_upsert
from src.LLM_answers import get_plan_using_LLM
from src.retrieve import retrieve_landmarks_images
from src.img_generation import generate_images
from src.evaluation import evaluate_retrieved_images, evaluate_generated_images, compare_results_Use_Case_1
from src.utils import get_start_time, get_end_time
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime


# system response pipeline
def get_RAG_response(request, text_index, id=None, eval=False):
    start_time = datetime.now()
    travel_plan, landmarks_list = get_plan_using_LLM(request)
    retrieved_images = retrieve_landmarks_images(text_index, landmarks_list)
    end_time = datetime.now()
    if eval:
        accuracy = evaluate_retrieved_images(retrieved_images, landmarks_list)
    else: 
        accuracy = None
    # save results
    results = {
        "id": id,
        "travel_plan": travel_plan,
        "landmarks_list": landmarks_list,
        "images": retrieved_images,
        "accuracy": accuracy,
        "start_time": start_time,
        "end_time": end_time,
        "response_by": "RAG",
        "use_case": "1"
    }
    return results


# baseline response pipeline
def get_baseline_response(request, id=None, eval=False):
    start_time = get_start_time()
    travel_plan, landmarks_list = get_plan_using_LLM(request)
    generated_imgs = generate_images(landmarks_list)
    end_time = get_end_time()
    if eval:
        accuracy = evaluate_generated_images(generated_imgs, landmarks_list)
    else:
        accuracy = None
    # save results
    results = {
        "id": id,
        "travel_plan": travel_plan,
        "landmarks_list": landmarks_list,
        "images": generated_imgs,
        "accuracy": accuracy,
        "start_time": start_time,
        "end_time": end_time,
        "response_by": "Generative Model",
        "use_case": "1"
    }
    return results


def load_user_requests():
    # TODO: there is already a fuction with the same name in src/data.py
    # Simulate loading user requests
    return ["(Venice) Doge's Palace and campanile of St. Mark's Basilica facing the sea.jpg",
            "Petřín Lookout Tower in Prague, 2012.jpg",
            "Rzym Fontanna piazza navona.jpg",
            # Add more file names as needed
            ], np.random.rand(50, 512)  # Random embeddings for testing


def load_and_embedd_dataset(rec_num=10):
    # TODO: move this function to src/data.py or tests folder
    # Simulate loading and embedding a dataset
    file_names = [
        "(Venice) Doge's Palace and campanile of St. Mark's Basilica facing the sea.jpg",
        "Petřín Lookout Tower in Prague, 2012.jpg",
        "Rzym Fontanna piazza navona.jpg",
        "St. Vitus Prague September 2016-21.jpg",
        "The Dancing House in Prague.jpg",
        # Add more file names as needed
    ]
    # Select a subset based on rec_num
    selected_names = file_names[:rec_num]
    # Generate random embeddings for the selected files
    embeddings = np.random.rand(rec_num, 512)  # Assuming embedding dimension is 512
    return selected_names, embeddings

def eval_pipeline_Use_Case_1():
    # TODO: implement comparison of RAG and baseline results for Use Case 1
    # User pipeline
    ids, requests = load_user_requests()
    # Prepare Data
    text_index = create_index_and_upsert(rec_num=50)

    all_RAG_results = []
    all_baseline_results = []
    for id, request, true_answer in zip(ids, requests):
        RAG_results = get_RAG_response(request, text_index, id, eval=True)
        baseline_results = get_baseline_response(request, id, eval=True)
        all_RAG_results.append(RAG_results)
        all_baseline_results.append(baseline_results)

    compare_results_Use_Case_1(all_RAG_results, all_baseline_results)

def inference_pipeline_Use_Case_1(query):
    # Prepare DB
    text_index = create_index_and_upsert(rec_num=50, embedding_model=SentenceTransformer('all-MiniLM-L6-v2'))
    # Get RAG response
    RAG_results = get_RAG_response(query, text_index)
    full_answer = RAG_results["full_answer"]
    retrieved_answer = RAG_results["retrieved_answer"]
    return full_answer, retrieved_answer

def test_pipeline():
    # Query Example
    query = "Plan a 2 week trip to Italy"
    # Get Results
    result = inference_pipeline_Use_Case_1(query)
    print(result)

if __name__ == "__main__":
    test_pipeline()
