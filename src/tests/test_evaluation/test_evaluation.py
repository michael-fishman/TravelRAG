import unittest
from src.evaluation import evaluate_landmark_name, evaluate_generated_images
import PIL.Image

class TestEvaluateLandmarkAnswer(unittest.TestCase):

    def test_correct_match(self):
        landmark_RAG_answer = "Eiffel Tower"
        true_answer = "Eiffel Tower"
        result = evaluate_landmark_name(landmark_RAG_answer, true_answer)
        self.assertEqual(result, "True")

    def test_correct_close_match(self):
        landmark_RAG_answer = "Notre Dame de Paris"
        true_answer = "Notre Dame Cathedral"
        result = evaluate_landmark_name(landmark_RAG_answer, true_answer)
        self.assertEqual(result, "True")

    def test_incorrect_match(self):
        landmark_RAG_answer = "Statue of Liberty"
        true_answer = "Eiffel Tower"
        result = evaluate_landmark_name(landmark_RAG_answer, true_answer)
        self.assertEqual(result, "False")


class TestEvaluateGeneratedImages(unittest.TestCase):
    
    def check_specific_set(self, generated_imgs, landmarks_list, expected_accuracy, expected_evaluation):
        # Call the function
        accuracy, evaluation = evaluate_generated_images(generated_imgs, landmarks_list)
        print(accuracy, expected_accuracy)
        print(f"accuracy: {accuracy} | expected_accuracy: {expected_accuracy}")
        print(f"bool_answers: {evaluation['bool_answers']} | expected_bool_answers: {expected_evaluation}")
        # Assertions
        self.assertAlmostEqual(accuracy, expected_accuracy) 
        self.assertEqual(evaluation['bool_answers'], expected_evaluation)
    
    def test_evaluate_generated_images(self):         
        # Test data
        Landmark_list = [
            'Brandenburg Gate, Berlin',
            'Eiffel Tower, Paris', 
            'Zannse Schans, Holland',
            'Big Ben, London'
        ]
        
        # Images Set No. 1
        img_file_names_1 = [
            'Arc_De_Triomphg_gate_paris.jpg',
            'Louvre museum.jpg',
            'Zaandam_holland.webp',
            'London-BigBen.jpeg',
        ]
        expected_evaluation_1 = ['no', 'no', 'no', 'yes']
        expected_accuracy_1 = 1 / 4
        img_paths_1 = [f'src/tests/test_evaluation/{img}' for img in img_file_names_1]
        generated_imgs_1 = [PIL.Image.open(img) for img in img_paths_1]
        # Check
        self.check_specific_set(generated_imgs_1, Landmark_list, expected_accuracy_1, expected_evaluation_1)
        
        # Images Set No. 2
        img_file_names_2 = [
            'brandenburg_gate.jpeg',
            'paris.jpeg',
            'zaanse-schans-holland.jpg',
            'tower_with_clock.webp'
        ]
        expected_evaluation_2 = ['yes', 'yes', 'yes', 'no']
        expected_accuracy_2 = 3 / 4
        img_paths_2 = [f'src/tests/test_evaluation/{img}' for img in img_file_names_2]
        generated_imgs_2 = [PIL.Image.open(img) for img in img_paths_2]
        # Check
        self.check_specific_set(generated_imgs_2, Landmark_list, expected_accuracy_2, expected_evaluation_2)

