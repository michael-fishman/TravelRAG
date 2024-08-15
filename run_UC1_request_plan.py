from data import load_user_requests, load_img_text_dataset
from index import init_text_index
from prompts import get_travel_plan_prompt
from LLM_answers import get_plan_using_LLM, create_final_travel_plan
from retrieve import retrive_landmarks_images
from img_generation import generate_images
from evaluation import evaluate_retrieved_images, evaluate_generated_images, compare_responses
from utils import get_start_time, get_end_time
from time_comparison import save_time_results


# system response pipeline
def get_system_response(request, text_index):
    start_time = get_start_time()
    prompt = get_travel_plan_prompt(request)
    travel_plan, landmarks_list = get_plan_using_LLM(prompt)
    retrieved_images = retrive_landmarks_images(text_index, landmarks_list)
    final_travel_plan = create_final_travel_plan(travel_plan, retrieved_images)
    end_time = get_end_time()
    save_time_results(start_time, end_time)
    evaluate_retrieved_images(retrieved_images, landmarks_list)
    return final_travel_plan


# baseline response pipeline
def get_baseline_response(request):
    start_time = get_start_time()
    prompt = get_travel_plan_prompt(request)
    travel_plan, landmarks_list = get_plan_using_LLM(prompt)
    generated_imgs = generate_images(landmarks_list)
    final_travel_plan = create_final_travel_plan(travel_plan, generated_imgs)
    end_time = get_end_time()
    save_time_results(start_time, end_time)
    evaluate_generated_images(generated_imgs, landmarks_list)
    return final_travel_plan


# User pipeline
requests = load_user_requests()
# Prepare Data
img_text_dataset = load_img_text_dataset()
# prepare DB
text_index = init_text_index(img_text_dataset)

for request in requests:
    system_response = get_system_response(request, text_index)
    baseline_response = get_baseline_response(request)
    compare_responses(system_response, baseline_response)
