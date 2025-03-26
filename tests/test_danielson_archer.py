import unittest
import sys
import os
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Import the application
from gradio_display.app import DanielsonArcherApp, DANIELSON_COMPONENTS, SAMPLE_LOW_INFERENCE_NOTES
from archer.backwardPass.danielson_model import DanielsonModel

class TestDanielsonArcher(unittest.TestCase):
    """
    Test cases for the Danielson-Archer implementation.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Mock the ArgillaDatabase
        self.mock_db = MagicMock()
        self.mock_db.connect.return_value = True
        self.mock_db.initialize_datasets.return_value = True
        self.mock_db.store_generated_content.return_value = "test_output_id"
        self.mock_db.store_evaluation.return_value = True
        
        # Mock the Archer instance
        self.mock_archer = MagicMock()
        mock_evaluation = (MagicMock(), "Test content", {
            "score": 4,
            "feedback": "Test feedback",
            "improved_output": "Test perfect output"
        })
        self.mock_archer.run_forward_pass.return_value = [mock_evaluation]
        
        # Create the app instance with mocks
        self.app = DanielsonArcherApp(
            archer_instance=self.mock_archer,
            argilla_db=self.mock_db
        )
    
    def test_generate_input_data(self):
        """
        Test that the generate_input_data method returns valid inputs.
        """
        low_inference_notes, component_id = self.app.generate_input_data()
        
        # Check that the returned values are valid
        self.assertIn(low_inference_notes, SAMPLE_LOW_INFERENCE_NOTES)
        self.assertIn(component_id, DANIELSON_COMPONENTS)
    
    def test_generate_summary_with_archer(self):
        """
        Test that generate_summary properly calls Archer when available.
        """
        result = self.app.generate_summary("Test notes", "1a")
        
        # Check that Archer was called
        self.mock_archer.run_forward_pass.assert_called_once()
        
        # Check the result structure
        self.assertEqual(result["content"], "Test content")
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["feedback"], "Test feedback")
        self.assertEqual(result["perfect_output"], "Test perfect output")
    
    @patch('gradio_display.app.generate_single_component_evaluation')
    def test_generate_summary_fallback(self, mock_generate):
        """
        Test the fallback behavior when Archer is not available.
        """
        # Set up the app without Archer
        app = DanielsonArcherApp(
            archer_instance=None,
            argilla_db=self.mock_db
        )
        
        # Set up the mock to return a valid result
        mock_generate.return_value = {
            "summary": "Test summary",
            "score": 3
        }
        
        # Call the method
        result = app.generate_summary("Test notes", "1a")
        
        # Check that the fallback method was called
        mock_generate.assert_called_once_with("Test notes", "1a")
        
        # Check the result structure
        self.assertEqual(result["content"], "Test summary")
        self.assertEqual(result["score"], 3)
    
    def test_save_to_database(self):
        """
        Test that save_to_database correctly stores data.
        """
        success = self.app.save_to_database(
            "Test notes", "1a", "Test content", 4, "Test feedback", "Test perfect output"
        )
        
        # Check that the database methods were called
        self.mock_db.store_generated_content.assert_called_once()
        self.mock_db.store_evaluation.assert_called_once()
        
        # Check the return value
        self.assertTrue(success)
    
    def test_trigger_optimization(self):
        """
        Test that trigger_optimization correctly calls the Archer backward pass.
        """
        # Save the initial round
        initial_round = self.app.current_round
        
        # Call the method
        result = self.app.trigger_optimization()
        
        # Check that Archer was called
        self.mock_archer.run_backward_pass.assert_called_once()
        
        # Check that the round was incremented
        self.assertEqual(self.app.current_round, initial_round + 1)
        
        # Check the return value contains the new round
        self.assertIn(str(self.app.current_round), result)
    
    def test_trigger_optimization_no_archer(self):
        """
        Test the behavior when triggering optimization without Archer.
        """
        # Set up the app without Archer
        app = DanielsonArcherApp(
            archer_instance=None,
            argilla_db=self.mock_db
        )
        
        # Save the initial round
        initial_round = app.current_round
        
        # Call the method
        result = app.trigger_optimization()
        
        # Check that the round was incremented
        self.assertEqual(app.current_round, initial_round + 1)
        
        # Check the return value mentions simulation
        self.assertIn("Simulated optimization", result)


if __name__ == "__main__":
    unittest.main() 