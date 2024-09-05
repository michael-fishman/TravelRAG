from src.data import load_user_requests_Use_Case_2
from src.index import create_index_and_upsert
from src.embeddings import get_img_embeddings
from src.LLM_answers import get_landmark_answer_using_LLM, get_landmark_answer_using_RAG
from src.retrieve import retrieve_landmarks_names
from src.evaluation import evaluate_landmark_name, compare_results_Use_Case_2, save_results_Use_Case_2
from transformers import CLIPModel
from datetime import datetime
from PIL import Image
from pinecone import Pinecone


# system response pipeline
def get_RAG_response(img_query: Image.Image, img_index: Pinecone, true_answer: str = None, id: int = None, user_name: str = None, eval: bool = False):
    """
    Get response from RAG model.

    Args:
        img_query (Image.Image): the user's request for a travel plan.
        img_index (Pincone): Pincone index for image embeddings.
        true_answer (str, optional): the true answer. Defaults to None.
        id (int, optional): the request id. Defaults to None.
        user_name (str, optional): the user's name. Defaults to None.
        eval (bool, optional): wether to eval or not . Defaults to False.

    Returns:
        dict: the response from the RAG model.
    """
    start_time = datetime.now()
    embedded_query = get_img_embeddings([img_query])[0]
    retrieved_answer = retrieve_landmarks_names(img_index, embedded_query)
    full_answer, landmark_RAG_answer = get_landmark_answer_using_RAG(retrieved_answer, user_name)
    end_time = datetime.now()
    if eval:
        correct = evaluate_landmark_name(landmark_RAG_answer, true_answer)
    else:
        correct = None
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
def get_baseline_response(img_query: Image.Image, true_answer: str=None, user_name: str=None, id: int=None, eval: bool=False):
    """
    Get response from the baseline model.

    Args:
        img_query (Image.Image): the user's request for a travel plan.
        true_answer (str, optional): the true answer. Defaults to None.
        user_name (str, optional): the user's name. Defaults to None.
        id (int, optional): the request id. Defaults to None.
        eval (bool, optional): wether to eval or not . Defaults to False.

    Returns:
        dict: the response from the baseline model.
    """
    start_time = datetime.now()
    full_answer, landmark_LLM_answer = get_landmark_answer_using_LLM(img_query, user_name)
    end_time = datetime.now()
    if eval:
        correct = evaluate_landmark_name(landmark_LLM_answer, true_answer)
    else:
        correct = None
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
    ids, requests, true_answers = load_user_requests_Use_Case_2()

    # prepare DB
    img_index = create_index_and_upsert(is_text_index=False, rec_num=-1)

    for id, request, true_answer in zip(ids, requests, true_answers):
        RAG_results = get_RAG_response(request, img_index, true_answer, id, eval=True)
        baseline_results = get_baseline_response(request, true_answer, id, eval=True)
        save_results_Use_Case_2(RAG_results, baseline_results)

    # compare_results_Use_Case_2(all_RAG_results, all_baseline_results)
    # TODO: implement comparison of RAG and baseline results for Use Case 2

def inference_pipeline_Use_Case_2(img):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # Prepare DB
    img_index = create_index_and_upsert(is_text_index=False, rec_num=50, embedding_model=model)
    # Get RAG response
    RAG_results = get_RAG_response(img, img_index)
    full_answer = RAG_results["full_answer"]
    retrieved_answer = RAG_results["retrieved_answer"]
    return full_answer, retrieved_answer

def test_pipeline():
    # Query Example
    img = "C:\GitBash\Git\TravelRAG\datasets\images\(Venice)_Doge_s_Palace_and_campanile_of_St._Mark_s_Basilica_facing_the_sea.jpg"
    # Read image
    img = Image.open(img)
    # Get Results
    full_answer, retrieved_answer = inference_pipeline_Use_Case_2(img)
    print(f"Full Answer: {full_answer}")

if __name__ == "__main__":
    eval_pipeline_Use_Case_2()
    # test_pipeline()
