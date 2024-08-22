import unittest
from src.LLM_answers import get_landmark_answer_using_LLM, get_plan_using_LLM
import ast
import PIL.Image

class TestGetLandMarkAnswer(unittest.TestCase):
    
    def check_format_landmark_answer(self, result):
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
        
    def check_format_full_answer(self, result):
        # TODO: Implement this function
        # Check if the result is a string
        if not isinstance(result, str):
            return False

        return True

    def check_single_image(self, img_path, true_landmark, user_name):
        img_query = PIL.Image.open(img_path)
        full_answer, landmark_answer = get_landmark_answer_using_LLM(img_query, user_name)
        print(f"\nTrue: {true_landmark} | ", f"Pred: {landmark_answer}")
        self.assertTrue(self.check_format_landmark_answer(landmark_answer), f"Result is not in the correct format: {landmark_answer}")
        print(f"\nfull_answer: \n{full_answer} \n")
        self.assertTrue(self.check_format_full_answer(full_answer), f"Result is not in the correct format: {full_answer}")
        
    def test_get_landmark_answer_using_LLM(self):
        img_paths = ['src/tests/test_LLM_answers/paris.jpeg',
                        'src/tests/test_LLM_answers/zaanse-schans-holland.jpg',
                        'src/tests/test_LLM_answers/London-BigBen.jpeg']
        true_landmarks = ["paris", "zaanse-schans-holland", "London-BigBen"]
        user_names = ["Alice", "Bob", "None"]
        
        for img_path, true_landmark, user_name in zip(img_paths, true_landmarks, user_names):
            print(f"\nLandmark: {true_landmark} ------------------------------------")
            self.check_single_image(img_path, true_landmark, user_name)
            
class TestTravelPlanResponseFormat(unittest.TestCase):

    def check_specific_request(self, user_request):
        try:
            # Get the response from the model
            travel_plan, landmarks_list = get_plan_using_LLM(user_request)
            
            # If no exception is raised, print the travel plan
            print("\nTravel Plan:")
            for day, landmark, description in zip(travel_plan['days'], travel_plan['landmarks'], travel_plan['descriptions']):
                print(f"Day {day}:")
                print(f"  Landmark: {landmark}")
                print(f"  Description: {description}")
                print()
                
        except ValueError as e:
            self.fail(f"get_plan_using_LLM raised ValueError unexpectedly: {e}")

    def test_travel_plan_response_format(self):
        user_request_1 = "Give me a plan for a trip for 2 weeks to Amsterdam"
        user_request_2 = "I need a weekend getaway plan to Paris"
        user_request_3 = "Suggest a vacation itinerary for 5 days to Tokyo"
        
        for i, user_request in enumerate([user_request_1, user_request_2, user_request_3]):
            print(f"\nPlan No. {i+1} ------------------------------------")
            print(f"\nUser Request: {user_request}")
            self.check_specific_request(user_request)
