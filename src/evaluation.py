import os
import pandas as pd
from src.LLM_answers import GEMINI_KEY_PATH
import google.generativeai as genai

import json
from typing import Dict, List
import copy
from PIL import Image
import time

with open(GEMINI_KEY_PATH) as f:
    GEMINI_API_KEY = f.read().strip()
genai.configure(api_key=GEMINI_API_KEY)


def evaluate_landmark_name(predicted_answer: str, true_answer: str) -> str:
    """
    Evaluate whether the predicted landmark name matches the true landmark name using a language model.

    Args:
        predicted_answer (str): The predicted landmark name.
        true_answer (str): The true landmark name.

    Returns:
        str: The evaluation result, either "True" or "False
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    # Construct the prompt to ask the LLM to evaluate the match between the two landmarks
    prompt = (
        f"I have two landmark names. The first is the predicted landmark: '{predicted_answer}'. "
        f"The second is the true landmark: '{true_answer}'.\n\n"
        "Please evaluate whether the predicted landmark matches the true landmark. "
        "Respond with either 'True' or 'False'."
    )

    # Generate the evaluation using the LLM)
    llm_response = model.generate_content(prompt)
    time.sleep(5)
    evaluation = llm_response.text.replace("\n", "").strip()

    # Ensure the response is either "Correct" or "Incorrect"
    if "True" in evaluation:
        return "True"
    else:
        return "False"


def evaluate_retrieved_images(retrieved_names: List[str], landmarks_list: List[str]):
    """
    Evaluate the accuracy of the retrieved images based on the true landmark names.

    Args:
        retrieved_names (List[str]): List of retrieved landmark names.
        landmarks_list (List[str]): List of true landmark names.

    Returns:
        float: The accuracy of the retrieved images.
        Dict: A dictionay of evaluation results:
            evaluation = {
                'bool_answers': bool_answers,
                'question_landmarks': question_landmarks,
                'img_landmarks': img_landmarks,
                'explanations': explanations
            }
    """
    evaluation = []
    for retrieved_name, true_name in zip(retrieved_names, landmarks_list):
        correct = evaluate_landmark_name(retrieved_name, true_name)
        evaluation.append({"retrieved_name": retrieved_name, "true_name": true_name, "correct": correct})
    accuracy = sum(evaluation[i]["correct"] == "True" for i in range(len(evaluation))) / len(evaluation)
    return accuracy, evaluation


def evaluate_generated_images(generated_imgs: List[Image.Image], landmarks_list: List[str]):
    """
    Evaluate the accuracy of the generated images based on the true landmark names.

    Args:
        generated_imgs (List[Image.Image]): List of generated images.
        landmarks_list (List[str]): List of true landmark names.

    Returns:
        float: The accuracy of the generated images.
        Dict: A dictionay of evaluation results:
            evaluation = {
                'bool_answers': bool_answers,
                'question_landmarks': question_landmarks,
                'img_landmarks': img_landmarks,
                'explanations': explanations
            }
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    evaluations = []
    for landmark, img in zip(landmarks_list, generated_imgs):
        answer = model.generate_content([
            (
                f"Is this img from {landmark}? \n"
                f"- Answer in JSON format: "
                f'{{"question_landmark": <landmark in question>, "img_landmark": <img_landmark>, "Explanation": "<provide here explanation>", "bool_answer": "<yes/no>"}}\n'
                f"- Ensure the output is properly formatted as valid JSON, with no extraneous characters, code blocks, or newline escape sequences.\n"
                f"- Do not include any surrounding backticks or labels like 'json'. The output should be a clean, parsable JSON object.\n\n"
            ),
            img
        ]).text

        # answer = model.generate_content([
        #     img,
        #     (
        #         f"Is this img from {landmark}? \n"
        #         f"- Answer in JSON format: "
        #         f'{{"question_landmark": <landmark in question>, "img_landmark": <img_landmark>, "Explanation": "<provide here explanation>", "bool_answer": "<yes/no>"}}\n'
        #         f"- Ensure the output is properly formatted as valid JSON, with no extraneous characters, code blocks, or newline escape sequences.\n"
        #         f"- Do not include any surrounding backticks or labels like 'json'. The output should be a clean, parsable JSON object.\n\n"
        #     )
        # ]).text
        time.sleep(5)
        evaluations.append(answer)

    # Extract answers from the evaluations
    bool_answers = []
    question_landmarks = []
    img_landmarks = []
    explanations = []
    for evaluation in evaluations:
        try:
            # Parse the JSON string into a Python dictionary
            eval_dict = json.loads(evaluation)
            # Extract the 'bool_answer' value and add to the list
            bool_answers.append(eval_dict['bool_answer'].lower())
            question_landmarks.append(eval_dict['question_landmark'])
            img_landmarks.append(eval_dict['img_landmark'])
            explanations.append(eval_dict['Explanation'])
        except json.JSONDecodeError:
            print(f"Failed to parse evaluation: {evaluation}")
    evaluation = {
        'bool_answers': bool_answers,
        'question_landmarks': question_landmarks,
        'img_landmarks': img_landmarks,
        'explanations': explanations
    }

    # Calculate accuracy
    accuracy = sum(answer.lower() == 'yes' for answer in bool_answers) / len(landmarks_list)
    return accuracy, evaluation


def compare_results_Use_Case_1(RAG_results: Dict, baseline_results: Dict):
    # TODO: this is a draft code generated by the AI, please complete it
    # Create a comparison dictionary
    comparison = {
        "id": [],
        "RAG_time": [],
        "Baseline_time": [],
        "Time_difference": [],
        "RAG_accuracy": [],
        "Baseline_accuracy": [],
        "Accuracy_difference": []
    }

    # Iterate over results to compute time and accuracy differences
    for rag, baseline in zip(RAG_results, baseline_results):
        comparison["id"].append(rag["id"])
        rag_time = pd.to_datetime(rag["end_time"]) - pd.to_datetime(rag["start_time"])
        baseline_time = pd.to_datetime(baseline["end_time"]) - pd.to_datetime(baseline["start_time"])
        comparison["RAG_time"].append(rag_time.total_seconds())
        comparison["Baseline_time"].append(baseline_time.total_seconds())
        comparison["Time_difference"].append((baseline_time - rag_time).total_seconds())

        comparison["RAG_accuracy"].append(rag["accuracy"])
        comparison["Baseline_accuracy"].append(baseline["accuracy"])
        comparison["Accuracy_difference"].append(rag["accuracy"] - baseline["accuracy"])

    # Convert to DataFrame for better visualization
    comparison_df = pd.DataFrame(comparison)
    return comparison_df


def compare_results_Use_Case_2(RAG_results: Dict, baseline_results: Dict):
    # TODO: complete
    raise NotImplementedError


def save_results_Use_Case_1(RAG_results: Dict, baseline_results: Dict,
                            results_dir="./eval_results/UseCase1_travel_plans"):
    """
    Save the results of the Use Case 1 evaluation to a df.
        results dict example:
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
    The images are saved in a separate folder under the results folder with the id name.

    Args:
        RAG_results (Dict): Dictionary containing RAG results.
        baseline_results (Dict): Dictionary containing baseline results.
        results_dir (str): Directory where results should be saved.

    Returns:
        None
    """
    results_file = os.path.join(results_dir, "results.csv")

    # Create deep copies of the input dictionaries to avoid modifying the originals
    RAG_results_copy = copy.deepcopy(RAG_results)
    baseline_results_copy = copy.deepcopy(baseline_results)

    # Save images to the corresponding id folder
    for result in [RAG_results_copy, baseline_results_copy]:
        images_folder = os.path.join(results_dir, str(int(result['id']) + 30))
        os.makedirs(images_folder, exist_ok=True)
        for i, img in enumerate(result['images']):
            img_filename = f"image_{i + 1}.jpg"
            img_path = os.path.join(images_folder, img_filename)
            img.save(img_path)  # Save the PIL image object

        # Remove the images from the result dictionary
        result.pop('images', None)

    # Create a DataFrame
    rag_df = pd.DataFrame([RAG_results_copy])
    baseline_df = pd.DataFrame([baseline_results_copy])
    combined_df = pd.concat([rag_df, baseline_df], ignore_index=True)

    # Check if the results file already exists
    if not os.path.isfile(results_file):
        # If the file does not exist, create it with the current results
        combined_df.to_csv(results_file, index=False)
    else:
        # If the file exists, append the new results
        combined_df.to_csv(results_file, mode='a', header=False, index=False)

    print(f"Results saved to {results_file}")


def save_results_Use_Case_2(RAG_results: Dict, baseline_results: Dict,
                            results_dir="./eval_results/UseCase2_location_identifier"):
    """
    Save the results of the Use Case 2 evaluation to a df.
        results dict example:
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

    Args:
        RAG_results Dict
        baseline_results Dict

    Returns:
        _type_: _description_
    """
    results_file = os.path.join(results_dir, "results.csv")

    # Create a DataFrame
    rag_df = pd.DataFrame([RAG_results])
    baseline_df = pd.DataFrame([baseline_results])
    combined_df = pd.concat([rag_df, baseline_df], ignore_index=True)

    # Check if the results file already exists
    if not os.path.isfile(results_file):
        # If the file does not exist, create it with the current results
        combined_df.to_csv(results_file, index=False)
    else:
        # If the file exists, append the new results
        combined_df.to_csv(results_file, mode='a', header=False, index=False)

    print(f"Results saved to {results_file}")
