from src.data import load_user_requests
from src.index import create_index_and_upsert
from src.prompts import get_travel_plan_prompt
from src.LLM_answers import get_plan_using_LLM, create_final_travel_plan
from src.retrieve import retrive_landmarks_images
from src.img_generation import generate_images
from src.evaluation import evaluate_retrieved_images, evaluate_generated_images, compare_results_Use_Case_1
from src.utils import get_start_time, get_end_time
import numpy as np
from random import random
from pinecone import QueryResponse


# system response pipeline
def get_RAG_response(request, text_index):
    start_time = get_start_time()
    prompt = get_travel_plan_prompt(request)
    travel_plan, landmarks_list = get_plan_using_LLM(prompt)
    retrieved_images = retrive_landmarks_images(text_index, landmarks_list)
    final_travel_plan = create_final_travel_plan(travel_plan, retrieved_images)
    end_time = get_end_time()
    accuracy = evaluate_retrieved_images(retrieved_images, landmarks_list)
    # save results
    results = {
        "id": id,
        "final_travel_plan": final_travel_plan,
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
def get_baseline_response(request):
    start_time = get_start_time()
    prompt = get_travel_plan_prompt(request)
    travel_plan, landmarks_list = get_plan_using_LLM(prompt)
    generated_imgs = generate_images(landmarks_list)
    final_travel_plan = create_final_travel_plan(travel_plan, generated_imgs)
    end_time = get_end_time()
    accuracy = evaluate_generated_images(generated_imgs, landmarks_list)
    # save results
    results = {
        "id": id,
        "final_travel_plan": final_travel_plan,
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
    # Simulate loading user requests
    return ["(Venice) Doge's Palace and campanile of St. Mark's Basilica facing the sea.jpg",
            "Petřín Lookout Tower in Prague, 2012.jpg",
            "Rzym Fontanna piazza navona.jpg",
            # Add more file names as needed
            ], np.random.rand(50, 512)  # Random embeddings for testing


def load_and_embedd_dataset(rec_num=10):
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


def run_full_pipeline_Use_Case_1():
    # User pipeline
    ids, requests = load_user_requests()
    # Prepare Data
    text_index = create_index_and_upsert(rec_num=50)

    all_RAG_results = []
    all_baseline_results = []
    for id, request, true_answer in zip(ids, requests):
        RAG_results = get_RAG_response(request, text_index, id)
        baseline_results = get_baseline_response(request, id)
        all_RAG_results.append(RAG_results)
        all_baseline_results.append(baseline_results)

    compare_results_Use_Case_1(all_RAG_results, all_baseline_results)


def test_pipeline():
    # Initialize and upsert data to the index
    index_upserted = create_index_and_upsert(rec_num=50)

    # Simulate a query to the Pinecone index
    query_embedding = [random() for i in range(384)]  # Random query embedding for testing
    query_result: QueryResponse = index_upserted.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # Print the query results
    print("Query results:")
    for match in query_result.matches:
        print(f"ID: {match.id}, Score: {match.score}, Metadata: {match.metadata}")


if __name__ == "__main__":
    test_pipeline()
