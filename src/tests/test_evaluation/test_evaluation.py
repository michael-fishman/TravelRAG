import unittest
from src.evaluation import evaluate_landmark_answer

class TestEvaluateLandmarkAnswer(unittest.TestCase):
    
    def test_correct_match(self):
        landmark_RAG_answer = "Eiffel Tower"
        true_answer = "Eiffel Tower"
        result = evaluate_landmark_answer(landmark_RAG_answer, true_answer)
        self.assertEqual(result, "Correct")
    
    def test_correct_close_match(self):
        landmark_RAG_answer = "Notre Dame de Paris"
        true_answer = "Notre Dame Cathedral"
        result = evaluate_landmark_answer(landmark_RAG_answer, true_answer)
        self.assertEqual(result, "Correct")
    
    def test_incorrect_match(self):
        landmark_RAG_answer = "Statue of Liberty"
        true_answer = "Eiffel Tower"
        result = evaluate_landmark_answer(landmark_RAG_answer, true_answer)
        self.assertEqual(result, "Incorrect")
