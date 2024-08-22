import unittest
from src.LLM_answers import get_landmark_answer_using_LLM
import ast

class TestGetLandMarkAnswer(unittest.TestCase):
    
    def check_format(self, result):
        try:
            # Attempt to parse the result as a dictionary
            parsed_result = ast.literal_eval(result)
            
            # Check if the result is a dictionary with the correct keys
            if not isinstance(parsed_result, dict) or 'name' not in parsed_result or 'location' not in parsed_result:
                return False
            
            landmark_name = parsed_result['name']
            landmark_location = parsed_result['location']
            # Check that the lengths of the landmark name and location are reasonable
            if len(landmark_name) > 100 or len(landmark_location) > 100:
                return False
            return True
        except (ValueError, SyntaxError):
            return False

    def test_get_landmark_answer_using_LLM(self):
        import PIL.Image
        img_query = PIL.Image.open('src/tests/test_LLM_answers/paris.jpeg')
        true_result = "paris"
        result = get_landmark_answer_using_LLM(img_query)
        print(f"\nTrue: {true_result} | ", f"Pred: {result}")
        self.assertTrue(self.check_format(result), f"Result is not in the correct format: {result}")
        
        img_query = PIL.Image.open('src/tests/test_LLM_answers/zaanse-schans-holland.jpg')
        true_result = "zaanse-schans-holland"
        result = get_landmark_answer_using_LLM(img_query)
        print(f"\nTrue: {true_result} | ", f"Pred: {result}")
        self.assertTrue(self.check_format(result), f"Result is not in the correct format: {result}")
        
        img_query = PIL.Image.open('src/tests/test_LLM_answers/London-BigBen.jpeg')
        true_result = "London-BigBen"
        result = get_landmark_answer_using_LLM(img_query)
        print(f"\nTrue: {true_result} | ", f"Pred: {result}")
        self.assertTrue(self.check_format(result), f"Result is not in the correct format: {result}")
