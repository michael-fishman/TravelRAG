from data import get_true_images

def evaluate_retrieved_images(retrieved_images, landmarks_list):
    # TODO: complete
    true_images = get_true_images(landmarks_list)
    raise NotImplementedError

def evaluate_generated_images(generated_imgs, landmarks_list):
    # TODO: complete
    true_images = get_true_images(landmarks_list)
    raise NotImplementedError

def evaluate_retrieved_images(landmark_RAG_answer, true_answer):
    # TODO: complete
    raise NotImplementedError

def evaluate_landmark_answer(landmark_RAG_answer, true_answer):
    # TODO: complete
    # Call for an LLM to evaluate the answer
    
    raise NotImplementedError

def compare_results_Use_Case_1(RAG_results, baseline_results):
    # TODO: complete
    raise NotImplementedError

def compare_results_Use_Case_2(RAG_results, baseline_results):
    # TODO: complete
    raise NotImplementedError