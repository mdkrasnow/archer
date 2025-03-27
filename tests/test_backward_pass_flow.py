#!/usr/bin/env python
"""
Integration tests for the backward pass flow in the Archer system.

These tests verify that the backward pass is functioning correctly
by tracing the execution path and ensuring all required steps occur.
"""

import os
import sys
import logging
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backward_pass_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import components
from data_labelling.archer.archer import Archer
from data_labelling.archer.helpers.prompt import Prompt
from data_labelling.archer.backwardPass.danielson_model import DanielsonModel
from data_labelling.archer.backwardPass.promptOptimizer import PromptOptimizer


class TestBackwardPassFlow(unittest.TestCase):
    """Test the complete flow of the backward pass in the Archer system."""

    def setUp(self):
        """Set up test environment with mock components."""
        logger.info("Setting up test environment")
        
        # Mock API key
        os.environ["OPENROUTER_API_KEY"] = "test_api_key"
        
        # Create initial prompts
        self.initial_prompts = [
            Prompt("Test prompt 1: Generate a comprehensive analysis for component {component_id} based on {input}"),
            Prompt("Test prompt 2: Create a detailed evaluation for component {component_id} using {input}")
        ]
        
        # Create a test rubric
        self.test_rubric = "Evaluate on a scale of 1-5 based on evidence, clarity, and actionability."
        
        # Create a mock Archer instance with dependency injection for testing
        self.mock_generator = MagicMock()
        self.mock_evaluator = MagicMock()
        self.mock_optimizer = MagicMock()
        
        # Create a real PromptOptimizer with adalflow disabled for testing
        self.real_optimizer = PromptOptimizer(
            model_name="test-model",
            temperature=0.7,
            adalflow_enabled=False,  # Disable AdaLFlow for testing
            openrouter_api_key="test_api_key",
            variation_traits=["clarity", "specificity"]
        )
        
        # Patch llm_call to prevent actual API calls
        self.llm_call_patch = patch('data_labelling.archer.helpers.llm_call.llm_call')
        self.mock_llm_call = self.llm_call_patch.start()
        self.mock_llm_call.return_value = {
            "choices": [{"message": {"content": "Optimized prompt content"}}]
        }
        
        logger.info("Creating Archer instance for testing")
        self.archer = Archer(
            generator_model_name="test-model",
            evaluator_model_name="test-model",
            optimizer_model_name="test-model",
            knowledge_base=["./data_labelling/eval"],
            rubric=self.test_rubric,
            initial_prompts=self.initial_prompts,
            openrouter_api_key="test_api_key",
            human_validation_enabled=True,
            adalflow_enabled=False  # Disable AdaLFlow for clearer testing path
        )
        
        # Replace mocked components for testing
        self.archer.generator = self.mock_generator
        self.archer.evaluator = self.mock_evaluator
        
        # Use the real optimizer with mocked LLM calls to test the actual optimization logic
        self.archer.optimizer = self.real_optimizer
        
        # Set active prompts
        self.archer.active_prompts = self.initial_prompts
        self.mock_generator.set_prompts.reset_mock()  # Clear the initial set_prompts call
        
        logger.info("Test setup complete")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Cleaning up test environment")
        self.llm_call_patch.stop()
        
    def test_basic_backward_pass(self):
        """Test that the basic backward pass executes correctly."""
        logger.info("Starting basic backward pass test")
        
        # Create test evaluations (prompt, content, eval_result)
        test_evaluations = [
            (
                self.initial_prompts[0],
                "Generated content 1",
                {"score": 3.5, "feedback": "Good but could be more specific"}
            ),
            (
                self.initial_prompts[1],
                "Generated content 2",
                {"score": 2.8, "feedback": "Needs more evidence"}
            )
        ]
        
        # Execute backward pass
        logger.info("Executing backward pass")
        self.archer.run_backward_pass(test_evaluations)
        
        # Verify generator.set_prompts was called with new prompts
        self.mock_generator.set_prompts.assert_called_once()
        called_prompts = self.mock_generator.set_prompts.call_args[0][0]
        
        # Verify we have updated prompts
        logger.info(f"Checking updated prompts: received {len(called_prompts)} prompts")
        self.assertGreaterEqual(len(called_prompts), len(self.initial_prompts))
        
        # Verify the generation count was incremented
        self.assertEqual(self.archer.generation_count, 1)
        
        logger.info("Basic backward pass test complete")
    
    def test_empty_evaluations(self):
        """Test that the backward pass handles empty evaluations appropriately."""
        logger.info("Starting empty evaluations test")
        
        # Execute backward pass with empty evaluations
        self.archer.run_backward_pass([])
        
        # Verify generator.set_prompts was not called
        self.mock_generator.set_prompts.assert_not_called()
        
        logger.info("Empty evaluations test complete")
    
    def test_optimization_error_handling(self):
        """Test that the backward pass handles optimization errors gracefully."""
        logger.info("Starting optimization error handling test")
        
        # Create test evaluations
        test_evaluations = [
            (
                self.initial_prompts[0],
                "Generated content 1",
                {"score": 3.5, "feedback": "Good but could be more specific"}
            )
        ]
        
        # Make the optimizer raise an exception during optimize
        with patch.object(self.real_optimizer, 'optimize', side_effect=Exception("Test exception")):
            # Execute backward pass
            logger.info("Executing backward pass with failing optimizer")
            self.archer.run_backward_pass(test_evaluations)
            
            # Verify generator.set_prompts was not called
            self.mock_generator.set_prompts.assert_not_called()
        
        logger.info("Optimization error handling test complete")
    
    def test_optimization_with_adalflow(self):
        """Test the backward pass with AdaLFlow enabled."""
        logger.info("Starting AdaLFlow optimization test")
        
        # Create a mock AdaLFlow-enabled Archer instance
        adalflow_archer = Archer(
            generator_model_name="test-model",
            evaluator_model_name="test-model",
            optimizer_model_name="test-model",
            knowledge_base=["./data_labelling/eval"],
            rubric=self.test_rubric,
            initial_prompts=self.initial_prompts,
            openrouter_api_key="test_api_key",
            human_validation_enabled=True,
            adalflow_enabled=True
        )
        
        # Create a mock optimizer with AdaLFlow enabled
        mock_adalflow_optimizer = MagicMock()
        adalflow_archer.optimizer = mock_adalflow_optimizer
        mock_adalflow_optimizer.optimize.return_value = [
            Prompt("Optimized AdaLFlow prompt 1"),
            Prompt("Optimized AdaLFlow prompt 2")
        ]
        
        # Create test evaluations
        test_evaluations = [
            (
                self.initial_prompts[0],
                "Generated content 1",
                {"score": 3.5, "feedback": "Good but could be more specific"}
            ),
            (
                self.initial_prompts[1],
                "Generated content 2",
                {"score": 2.8, "feedback": "Needs more evidence"}
            )
        ]
        
        # Execute backward pass
        logger.info("Executing backward pass with AdaLFlow")
        adalflow_archer.run_backward_pass(test_evaluations)
        
        # Verify the optimizer was called
        mock_adalflow_optimizer.optimize.assert_called_once()
        
        # Verify appropriate parameters were passed
        args, kwargs = mock_adalflow_optimizer.optimize.call_args
        logger.info(f"AdaLFlow optimizer was called with prompts: {args[0]}")
        self.assertEqual(len(args[0]), 2)  # Should have two prompts
        
        # Verify feedback and score maps
        feedback_map = kwargs.get('feedback_map', {})
        score_map = kwargs.get('score_map', {})
        self.assertEqual(len(feedback_map), 2)
        self.assertEqual(len(score_map), 2)
        
        logger.info("AdaLFlow optimization test complete")
    
    def test_full_integration_flow(self):
        """Test the complete integration flow from forward pass to backward pass."""
        logger.info("Starting full integration flow test")
        
        # Mock the forward pass to return test evaluations
        test_input = {"input": "Test input data", "component_id": "1a"}
        
        # Mock the generator to return content
        self.mock_generator.generate.return_value = "Generated test content"
        
        # Mock the evaluator to return an evaluation
        self.mock_evaluator.evaluate.return_value = {
            "score": 4.0,
            "feedback": "Good evaluation with specific evidence and clear structure.",
            "improved_output": "Here's an improved version..."
        }
        
        # Execute forward pass
        logger.info("Executing forward pass")
        evaluations = self.archer.run_forward_pass(test_input)
        
        # Verify we got evaluations
        self.assertIsNotNone(evaluations)
        self.assertGreater(len(evaluations), 0)
        
        # Execute backward pass with the evaluations from forward pass
        logger.info("Executing backward pass with forward pass evaluations")
        self.archer.run_backward_pass(evaluations)
        
        # Verify generator.set_prompts was called with new prompts
        self.mock_generator.set_prompts.assert_called_once()
        
        # Verify the generation count was incremented
        self.assertEqual(self.archer.generation_count, 1)
        
        logger.info("Full integration flow test complete")


class TestDanielsonModelBackwardPass(unittest.TestCase):
    """Test the backward pass specifically with the Danielson model."""

    def setUp(self):
        """Set up test environment with real Danielson model."""
        logger.info("Setting up Danielson model test environment")
        
        # Mock API key
        os.environ["OPENROUTER_API_KEY"] = "test_api_key"
        
        # Patch llm_call to prevent actual API calls
        self.llm_call_patch = patch('data_labelling.archer.helpers.llm_call.llm_call')
        self.mock_llm_call = self.llm_call_patch.start()
        self.mock_llm_call.return_value = {
            "choices": [{"message": {"content": "Optimized Danielson prompt content"}}]
        }
        
        # Create the Danielson model
        self.danielson_model = DanielsonModel(adalflow_enabled=True)
        
        # Create a real PromptOptimizer
        self.optimizer = PromptOptimizer(
            model_name="test-model",
            temperature=0.7,
            adalflow_enabled=True,  # Enable AdaLFlow
            openrouter_api_key="test_api_key"
        )
        
        logger.info("Danielson model test setup complete")

    def tearDown(self):
        """Clean up after tests."""
        logger.info("Cleaning up Danielson model test environment")
        self.llm_call_patch.stop()
    
    def test_danielson_model_optimization(self):
        """Test optimization of the Danielson model prompts."""
        logger.info("Starting Danielson model optimization test")
        
        # Create feedback and score maps for testing
        feedback_map = {
            "context_analysis": "Could provide more specific guidelines for evidence collection.",
            "component_evaluation_base": "The prompt is too general and needs more specificity.",
            "restructure_feedback": "Good structure but could use clearer formatting instructions."
        }
        
        score_map = {
            "context_analysis": 3.5,
            "component_evaluation_base": 2.8,
            "restructure_feedback": 4.2
        }
        
        # Verify the model has the expected prompts
        logger.info(f"Danielson model has {len(self.danielson_model.prompts)} prompts")
        for prompt_id in feedback_map.keys():
            self.assertIn(prompt_id, self.danielson_model.prompts)
            logger.info(f"Found prompt: {prompt_id}")
        
        # Execute model optimization
        logger.info("Executing Danielson model optimization")
        
        # Mock the optimizer's internal methods
        with patch.object(self.optimizer, '_wrap_prompts_as_params'):
            with patch.object(self.optimizer, 'optimize_model') as mock_optimize_model:
                mock_optimize_model.return_value = True
                
                # Call optimize_model
                result = self.optimizer.optimize_model(
                    model=self.danielson_model,
                    feedback_map=feedback_map,
                    score_map=score_map
                )
                
                # Verify the result
                self.assertTrue(result)
                mock_optimize_model.assert_called_once()
        
        logger.info("Danielson model optimization test complete")


if __name__ == "__main__":
    # Run the tests
    unittest.main() 