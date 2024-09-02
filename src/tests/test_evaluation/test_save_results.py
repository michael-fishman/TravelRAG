import unittest
import os
import pandas as pd
from tempfile import TemporaryDirectory
from src.evaluation import save_results_Use_Case_2, save_results_Use_Case_1
from PIL import Image

class TestSaveResultsUseCase2(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory to store the test files
        self.test_dir = TemporaryDirectory()
        self.results_dir = os.path.join(self.test_dir.name, "eval_results/UseCase2_location_identifier")
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_file = os.path.join(self.results_dir, "results.csv")
        
        # Example test data
        self.RAG_results = {
            "id": 1,
            "full_answer": "This is a full answer.",
            "answer": "landmark_LLM_answer",
            "retrieved_answer": None,
            "true_answer": "true_answer",
            "correct": True,
            "start_time": "2024-01-01 10:00:00",
            "end_time": "2024-01-01 10:01:00",
            "response_by": "Generative Model",
            "use_case": "2"
        }
        
        self.baseline_results = {
            "id": 1,
            "full_answer": "This is a full baseline answer.",
            "answer": "baseline_LLM_answer",
            "retrieved_answer": None,
            "true_answer": "true_answer",
            "correct": False,
            "start_time": "2024-01-01 10:00:00",
            "end_time": "2024-01-01 10:01:00",
            "response_by": "Baseline Model",
            "use_case": "2"
        }
    
    def tearDown(self):
        # Clean up the temporary directory after the test
        self.test_dir.cleanup()

    def test_save_results_creates_file(self):
        # Test that the function creates the file
        save_results_Use_Case_2(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        self.assertTrue(os.path.isfile(self.results_file), "The results file was not created.")
    
    def test_save_results_appends_data(self):
        # Test that the function appends data correctly
        save_results_Use_Case_2(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        save_results_Use_Case_2(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        
        # Load the results file into a DataFrame and check its length
        df = pd.read_csv(self.results_file)
        self.assertEqual(len(df), 4, "The results file does not contain the correct number of rows.")

    def test_save_results_correct_data(self):
        # Test that the correct data is saved
        save_results_Use_Case_2(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        
        # Load the results file into a DataFrame and check its contents
        df = pd.read_csv(self.results_file)
        self.assertTrue("full_answer" in df.columns, "The results file does not contain the correct columns.")
        self.assertEqual(df.iloc[0]['full_answer'], self.RAG_results['full_answer'], "The saved data is incorrect.")


class TestSaveResultsUseCase1(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory to store the test files
        self.test_dir = TemporaryDirectory()
        self.results_dir = os.path.join(self.test_dir.name, "eval_results/UseCase1_travel_plans")
        os.makedirs(self.results_dir, exist_ok=True)
        self.results_file = os.path.join(self.results_dir, "results.csv")
        
        # Path to the test images provided in the src/tests/test_evaluation directory
        self.test_images_dir = os.path.join("src", "tests", "test_evaluation")
        
        # Load the images
        image1 = Image.open(os.path.join(self.test_images_dir, "Arc_De_Triomphg_gate_paris.jpg"))
        image2 = Image.open(os.path.join(self.test_images_dir, "brandenburg_gate.jpeg"))
        image3 = Image.open(os.path.join(self.test_images_dir, "London-BigBen.jpeg"))
        image4 = Image.open(os.path.join(self.test_images_dir, "Louvre museum.jpg"))

        # Example test data using the provided images
        self.RAG_results = {
            "id": 1,
            "travel_plan": "Plan A",
            "landmarks_list": ["Arc_De_Triomphg_gate_paris", "brandenburg_gate"],
            "images": [image1, image2],
            "accuracy": 0.9,
            "evaluation": {"key1": "value1", "key2": ["list_item1", "list_item2"]},
            "start_time": "2024-01-01 10:00:00",
            "end_time": "2024-01-01 10:01:00",
            "response_by": "RAG",
            "use_case": "1"
        }
        
        self.baseline_results = {
            "id": 2,
            "travel_plan": "Plan B",
            "landmarks_list": ["London-BigBen", "Louvre museum"],
            "images": [image3, image4],
            "accuracy": 0.8,
            "evaluation": {"key3": "value3", "key4": ["list_item3", "list_item4"]},
            "start_time": "2024-01-01 10:02:00",
            "end_time": "2024-01-01 10:03:00",
            "response_by": "Baseline",
            "use_case": "1"
        }
    
    def tearDown(self):
        # Clean up the temporary directory after the test
        self.test_dir.cleanup()

    def test_save_results_creates_file_and_saves_images(self):
        # Test that the function creates the file and saves the images
        save_results_Use_Case_1(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        
        # Check if the results CSV file was created
        self.assertTrue(os.path.isfile(self.results_file), "The results file was not created.")
        
        # Check if the images were saved to the correct folders
        rag_images_dir = os.path.join(self.results_dir, str(self.RAG_results['id']))
        baseline_images_dir = os.path.join(self.results_dir, str(self.baseline_results['id']))
        
        self.assertTrue(os.path.isdir(rag_images_dir), "RAG images folder was not created.")
        self.assertTrue(os.path.isdir(baseline_images_dir), "Baseline images folder was not created.")
        
        # Check if images were saved in the RAG and baseline folders
        self.assertTrue(os.path.isfile(os.path.join(rag_images_dir, "image_1.jpg")), "RAG image 1 was not saved.")
        self.assertTrue(os.path.isfile(os.path.join(rag_images_dir, "image_2.jpg")), "RAG image 2 was not saved.")
        self.assertTrue(os.path.isfile(os.path.join(baseline_images_dir, "image_1.jpg")), "Baseline image 1 was not saved.")
        self.assertTrue(os.path.isfile(os.path.join(baseline_images_dir, "image_2.jpg")), "Baseline image 2 was not saved.")
    
    def test_save_results_appends_data(self):
        # Test that the function appends data correctly
        save_results_Use_Case_1(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        save_results_Use_Case_1(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        
        # Load the results file into a DataFrame and check its length
        df = pd.read_csv(self.results_file)
        self.assertEqual(len(df), 4, "The results file does not contain the correct number of rows.")

    def test_save_results_correct_data(self):
        # Test that the correct data is saved
        save_results_Use_Case_1(self.RAG_results, self.baseline_results, results_dir=self.results_dir)
        
        # Load the results file into a DataFrame and check its contents
        df = pd.read_csv(self.results_file)
        self.assertTrue("travel_plan" in df.columns, "The results file does not contain the correct columns.")
        self.assertEqual(df.iloc[0]['travel_plan'], self.RAG_results['travel_plan'], "The saved data is incorrect.")
        self.assertEqual(df.iloc[0]['accuracy'], self.RAG_results['accuracy'], "The saved accuracy is incorrect.")
        
        # Test if the evaluation dictionary was saved correctly
        # Since evaluation is a dictionary, it should be stored as a string representation
        self.assertEqual(df.iloc[0]['evaluation'], str(self.RAG_results['evaluation']), "The saved evaluation data is incorrect.")


