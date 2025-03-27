"""
Tests for the refactored AIExpert evaluator with prompt tracking.
"""
import unittest
from unittest.mock import MagicMock, patch
from archer.forwardPass.evaluator import AIExpert

class TestEvaluatorRefactoring(unittest.TestCase):
    """Test the refactored AIExpert evaluator with prompt tracking."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the llm_call function
        self.patcher = patch('archer.forwardPass.evaluator.llm_call')
        self.mock_llm_call = self.patcher.start()
        
        # Set up mock LLM response
        self.mock_llm_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Score: 4\nFeedback: Good job\nImproved Output: Better output\nSummary: Nice work"
                    }
                }
            ]
        }
        
        # Create AIExpert instance
        self.evaluator = AIExpert(
            model_name="test-model",
            knowledge_base=["Test document"],
            rubric="Test rubric"
        )
    
    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
    
    def test_default_prompt_initialization(self):
        """Test that the evaluator initializes with a default prompt."""
        # Check that current_prompt is not empty
        self.assertTrue(self.evaluator.current_prompt)
        self.assertIn("You are an expert evaluator", self.evaluator.current_prompt)
        self.assertIn("Rubric: Test rubric", self.evaluator.current_prompt)
    
    def test_set_prompt(self):
        """Test setting a custom evaluator prompt."""
        custom_prompt = "Custom evaluator prompt with {input_placeholder} and {content_placeholder}"
        self.evaluator.set_prompt(custom_prompt)
        self.assertEqual(self.evaluator.current_prompt, custom_prompt)
    
    def test_get_current_prompt(self):
        """Test getting the current evaluator prompt."""
        self.evaluator.current_prompt = "Test prompt"
        self.assertEqual(self.evaluator.get_current_prompt(), "Test prompt")
    
    def test_evaluate_with_current_prompt(self):
        """Test that evaluate uses the current prompt with substituted values."""
        # Set a custom prompt with placeholders
        custom_prompt = "Evaluate: {input_placeholder} / {content_placeholder}"
        self.evaluator.set_prompt(custom_prompt)
        
        # Call evaluate
        self.evaluator.evaluate("Generated content", "Test input")
        
        # Verify that llm_call was called with the substituted prompt
        calls = self.mock_llm_call.call_args_list
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        messages = kwargs.get('messages', [])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Evaluate: Test input / Generated content") 