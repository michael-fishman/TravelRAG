# from src.data import get_true_images
from src.LLM_answers import GEMINI_KEY_PATH
import google.generativeai as genai
import json

with open(GEMINI_KEY_PATH) as f:
    GEMINI_API_KEY = f.read().strip()
genai.configure(api_key=GEMINI_API_KEY)


def evaluate_retrieved_images(retrieved_images, landmarks_list):
    # TODO: complete
    # true_images = get_true_images(landmarks_list)
    raise NotImplementedError

def evaluate_generated_images(generated_imgs, landmarks_list):
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

def evaluate_landmark_answer(landmark_RAG_answer, true_answer):    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    # Construct the prompt to ask the LLM to evaluate the match between the two landmarks
    prompt = (
        f"I have two landmark names. The first is the predicted landmark: '{landmark_RAG_answer}'. "
        f"The second is the true landmark: '{true_answer}'.\n\n"
        "Please evaluate whether the predicted landmark matches the true landmark. "
        "Respond with either 'True' or 'False'."
    )
    
    # Generate the evaluation using the LLM)
    llm_response = model.generate_content(prompt)
    evaluation = llm_response.text.replace("\n", "").strip()
    
    # Ensure the response is either "Correct" or "Incorrect"
    if "True" in evaluation:
        return "True"
    else:
        return "False"

def compare_results_Use_Case_1(RAG_results, baseline_results):
    # TODO: complete
    raise NotImplementedError

def compare_results_Use_Case_2(RAG_results, baseline_results):
    # TODO: complete
    raise NotImplementedError