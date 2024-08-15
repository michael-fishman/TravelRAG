from data import load_user_requests
from index import init_text_index
from prompts import get_travel_plan_prompt
from LLM_answers import get_plan_using_LLM, create_final_travel_plan
from retrieve import retrive_landmarks_images
from img_generation import generate_images
from evaluation import evaluate_retrieved_images, evaluate_generated_images, compare_results_Use_Case_1
from utils import get_start_time, get_end_time


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

if __name__ == "__main__":
    # User pipeline
    ids, requests = load_user_requests()
    # Prepare Data
    img_text_dataset = load_img_text_dataset()
    # prepare DB
    text_index = init_text_index(img_text_dataset)

    all_RAG_results = []
    all_baseline_results = []
    for id, request, true_answer in zip(ids, requests):
        RAG_results = get_RAG_response(request, text_index, id)
        baseline_results = get_baseline_response(request, id)
        all_RAG_results.append(RAG_results)
        all_baseline_results.append(baseline_results)

    compare_results_Use_Case_1(all_RAG_results, all_baseline_results)
