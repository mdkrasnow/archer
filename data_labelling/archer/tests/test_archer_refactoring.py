"""
Integration tests for the refactored Archer class with integrated prompts.
"""
import unittest
from unittest.mock import MagicMock, patch
from archer.archer import Archer
from archer.helpers.prompt import Prompt
from archer.forwardPass.generator import GenerativeModel
from archer.forwardPass.evaluator import AIExpert

class TestArcherRefactoring(unittest.TestCase):
    """Test the refactored Archer class with integrated prompts."""
    
    def setUp(self):
        """Set up test environment with mock components."""
        # Create mock components
        self.mock_generator = MagicMock(spec=GenerativeModel)
        self.mock_evaluator = MagicMock(spec=AIExpert)
        self.mock_optimizer = MagicMock()
        self.mock_database = MagicMock()
        
        # Patch the required components
        self.generator_patcher = patch('archer.archer.GenerativeModel', return_value=self.mock_generator)
        self.evaluator_patcher = patch('archer.archer.AIExpert', return_value=self.mock_evaluator)
        self.optimizer_patcher = patch('archer.archer.PromptOptimizer', return_value=self.mock_optimizer)
        
        # Start the patchers
        self.mock_generator_class = self.generator_patcher.start()
        self.mock_evaluator_class = self.evaluator_patcher.start()
        self.mock_optimizer_class = self.optimizer_patcher.start()
        
        # Create test prompts
        self.test_prompts = [
            Prompt(content="Test prompt 1", score=4.0),
            Prompt(content="Test prompt 2", score=3.5)
        ]
        
        # Create an Archer instance
        self.archer = Archer(
            generator_model_name="test-model",
            evaluator_model_name="test-model",
            optimizer_model_name="test-model",
            knowledge_base=[],
            rubric="Test rubric",
            initial_prompts=self.test_prompts,
            openrouter_api_key="test-key"
        )
        
        # Set the database explicitly
        self.archer.database = self.mock_database
    
    def tearDown(self):
        """Clean up patches."""
        self.generator_patcher.stop()
        self.evaluator_patcher.stop()
        self.optimizer_patcher.stop()
    
    def test_initialization_with_prompts(self):
        """Test that Archer initializes with the provided prompts."""
        # Check that prompts were stored directly in Archer
        self.assertEqual(len(self.archer.active_generator_prompts), 2)
        self.assertEqual(self.archer.active_generator_prompts[0].content, "Test prompt 1")
        self.assertEqual(self.archer.active_generator_prompts[1].content, "Test prompt 2")
        
        # Check that prompts were set in the generator
        self.mock_generator.set_prompts.assert_called_with(self.test_prompts)
    
    def test_forward_pass_with_integrated_prompts(self):
        """Test that forward pass uses the integrated prompt approach."""
        # Set up mock returns
        self.mock_generator.generate.return_value = [
            ("Generated content", self.test_prompts[0])
        ]
        self.mock_evaluator.evaluate.return_value = {
            'score': 4.2,
            'feedback': 'Test feedback',
            'improved_output': 'Improved content',
            'summary': 'Test summary'
        }
        self.mock_evaluator.get_current_prompt.return_value = "Evaluator prompt"
        
        # Run forward pass
        result = self.archer.run_forward_pass("Test input")
        
        # Verify generator was called with the active prompts
        self.mock_generator.set_prompts.assert_called_with(self.test_prompts[:self.archer.max_prompts_per_cycle])
        
        # Verify store_record was called with prompt information
        self.mock_database.store_record.assert_called_with(
            input_data="Test input",
            content="Generated content",
            generator_prompt=self.test_prompts[0].content,
            evaluator_prompt="Evaluator prompt",
            prompt_generation=self.test_prompts[0].generation,
            round_id="0"
        )
    
    def test_backward_pass_with_integrated_prompts(self):
        """Test that backward pass handles the integrated prompt structure."""
        # Set up evaluations
        evaluations = [
            (self.test_prompts[0], "Generated content 1", {'score': 4.2, 'feedback': 'Feedback 1'}),
            (self.test_prompts[1], "Generated content 2", {'score': 3.8, 'feedback': 'Feedback 2'})
        ]
        
        # Set up mock optimizer return
        optimized_prompts = [
            Prompt(content="Optimized prompt 1", score=4.5),
            Prompt(content="Optimized prompt 2", score=4.0)
        ]
        self.mock_optimizer.optimize.return_value = optimized_prompts
        
        # Mock the evaluate and select method
        self.archer._evaluate_and_select_best_prompts = MagicMock(return_value=optimized_prompts)
        
        # Run backward pass
        self.archer.run_backward_pass(evaluations)
        
        # Verify optimizer was called with the correct feedback and scores
        expected_feedback_map = {'0': 'Feedback 1', '1': 'Feedback 2'}
        expected_score_map = {'0': 4.2, '1': 3.8}
        self.mock_optimizer.optimize.assert_called_with(
            [self.test_prompts[0], self.test_prompts[1]],
            expected_feedback_map,
            expected_score_map
        )
        
        # Verify best prompts were selected and set
        self.archer._evaluate_and_select_best_prompts.assert_called_with(optimized_prompts)
        self.assertEqual(self.archer.active_generator_prompts, optimized_prompts)
        self.mock_generator.set_prompts.assert_called_with(optimized_prompts) 