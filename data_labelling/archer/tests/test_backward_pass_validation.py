"""
Test for the backward pass validation workflow.
This test ensures that evaluations are properly stored and retrieved 
for the backward pass optimization.
"""

import unittest
import logging
import uuid
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add the root directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data_labelling.archer.database.supabase import SupabaseDatabase
from data_labelling.archer.archer import Archer
from data_labelling.archer.helpers.prompt import Prompt
from data_labelling.gradio_display.app import DanielsonArcherApp


class TestBackwardPassValidation(unittest.TestCase):
    """Test the backward pass validation workflow."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Argilla client
        self.mock_client = MagicMock()
        
        # Mock datasets
        self.evaluations_dataset = MagicMock()
        self.outputs_dataset = MagicMock()
        
        # Configure mock database
        self.db = SupabaseDatabase(api_url="mock://url", api_key="mock_key")
        self.db.client = self.mock_client
        self.db.datasets = {
            "evaluations": self.evaluations_dataset,
            "outputs": self.outputs_dataset
        }
        
        # Mock User ID
        self.db.user_id = "test_user"
        
        # Set up mock for records
        self.evaluations_dataset.records = MagicMock()
        self.outputs_dataset.records = MagicMock()
        
        # Initialize Archer
        self.initial_prompts = [
            Prompt("Test prompt 1: {input}"),
            Prompt("Test prompt 2: {input}")
        ]
        
        self.archer = Archer(
            generator_model_name="test-model",
            evaluator_model_name="test-model",
            optimizer_model_name="test-model",
            knowledge_base=[],
            rubric="Test rubric",
            initial_prompts=self.initial_prompts,
            openrouter_api_key="test_key"
        )
        
        # Initialize app
        self.app = DanielsonArcherApp(
            archer_instance=self.archer,
            supabase_db=self.db
        )
    
    def test_save_evaluation(self):
        """Test saving an evaluation to the database."""
        # Set up mock for store_generated_content
        output_id = str(uuid.uuid4())
        self.db.store_generated_content = MagicMock(return_value=output_id)
        
        # Set up mock for _get_output
        mock_output = {
            "fields": {
                "input": "test input",
                "generated_content": "test content"
            },
            "metadata": {
                "prompt_id": "test_prompt_id"
            }
        }
        self.db._get_output = MagicMock(return_value=mock_output)
        
        # Set up mock for records.log
        self.evaluations_dataset.records.log = MagicMock(return_value=None)
        
        # Set up mock for _get_latest_evaluation
        self.db._get_latest_evaluation = MagicMock(return_value={
            "metadata": {
                "is_human": "1",
                "status": "submitted"
            }
        })
        
        # Save an evaluation
        result = self.app.save_to_database(
            low_inference_notes="Test notes",
            component_id="1a",
            content="Test content",
            score=4,
            feedback="Test feedback",
            perfect_output="Perfect output example"
        )
        
        # Verify results
        self.assertTrue(result)
        self.db.store_generated_content.assert_called_once()
        self.evaluations_dataset.records.log.assert_called_once()
        
        # Verify the correct metadata was set
        call_args = self.evaluations_dataset.records.log.call_args[0][0]
        self.assertEqual(len(call_args), 1)
        record = call_args[0]
        
        # Check metadata structure
        self.assertIn('metadata', record.__dict__)
        metadata = record.metadata
        self.assertIn('is_human', metadata)
        self.assertEqual(metadata['is_human'], '1')  # Should be marked as human
        self.assertIn('status', metadata)
        self.assertEqual(metadata['status'], 'submitted')  # Should be submitted
    
    def test_get_validated_evaluations(self):
        """Test retrieving validated evaluations."""
        # Create mock records with the proper format
        mock_records = [
            {
                "metadata": {
                    "output_id": "test1",
                    "prompt_id": "prompt1",
                    "is_human": "1",
                    "timestamp": datetime.now().isoformat()
                },
                "fields": {
                    "input": "test input 1",
                    "generated_content": "test content 1"
                },
                "responses": [
                    {"question": {"name": "score"}, "value": "4"},
                    {"question": {"name": "feedback"}, "value": "Good job"},
                    {"question": {"name": "improved_output"}, "value": "Better output"}
                ]
            },
            {
                "metadata": {
                    "output_id": "test2",
                    "prompt_id": "prompt2",
                    "is_human": "1",
                    "timestamp": datetime.now().isoformat()
                },
                "fields": {
                    "input": "test input 2",
                    "generated_content": "test content 2"
                },
                "responses": [
                    {"question": {"name": "score"}, "value": "5"},
                    {"question": {"name": "feedback"}, "value": "Excellent"},
                    {"question": {"name": "improved_output"}, "value": "Perfect output"}
                ]
            }
        ]
        
        # Set up mock for query
        mock_query_result = MagicMock()
        mock_query_result.to_list.return_value = mock_records
        self.evaluations_dataset.records.return_value = mock_query_result
        
        # Get validated evaluations
        result = self.db.get_validated_evaluations(limit=10)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIn('output_id', result.columns)
        self.assertIn('prompt_id', result.columns)
        self.assertIn('score', result.columns)
        self.assertIn('feedback', result.columns)
        self.assertIn('improved_output', result.columns)
    
    def test_backward_pass_with_validations(self):
        """Test the backward pass with validated evaluations."""
        # Mock get_validated_evaluations
        mock_evaluations = MagicMock()
        mock_evaluations.iterrows.return_value = [
            (0, {
                "output_id": "test1",
                "prompt_id": self.initial_prompts[0].id if hasattr(self.initial_prompts[0], 'id') else "prompt1",
                "generated_content": "test content 1",
                "input": json.dumps({"component_id": "1a", "input": "test input"}),
                "score": 4,
                "feedback": "Good job",
                "improved_output": "Better output"
            })
        ]
        self.db.get_validated_evaluations = MagicMock(return_value=mock_evaluations)
        
        # Original run_backward_pass
        original_run_backward_pass = self.archer.run_backward_pass
        self.archer.run_backward_pass = MagicMock()
        
        # Trigger backward pass
        result = self.app.trigger_backward_pass()
        
        # Verify results
        self.assertTrue(result)
        self.db.get_validated_evaluations.assert_called_once()
        self.archer.run_backward_pass.assert_called_once()
        
        # Restore original method
        self.archer.run_backward_pass = original_run_backward_pass
        
        # Check if active_prompts is properly handled
        # It should be an alias for active_generator_prompts
        self.assertEqual(self.archer.active_prompts, self.archer.active_generator_prompts)


if __name__ == '__main__':
    unittest.main() 