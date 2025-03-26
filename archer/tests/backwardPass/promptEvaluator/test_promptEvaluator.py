import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backwardPass.PromptEvaluator.promptEvaluator import PromptEvaluator
from helpers.prompt import Prompt
from forwardPass.evaluator import AIExpert
from forwardPass.generator import GenerativeModel

class TestPromptEvaluator:
    def test_prompt_evaluator_initialization(self):
        """Test that PromptEvaluator initializes with correct parameters"""
        # Create mock objects for dependencies
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        num_simulations = 5
        quantile_threshold = 0.3
        
        # Initialize the PromptEvaluator
        evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations, quantile_threshold)
        
        # Verify initialization
        assert evaluator.generative_model == generator_mock
        assert evaluator.evaluator == evaluator_mock
        assert evaluator.num_simulations == num_simulations
        assert evaluator.quantile_threshold == quantile_threshold
    
    def test_evaluate_prompts(self):
        """Test evaluation of prompts through simulated forward pass"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator mock to return generated content
        generator_mock.generate.return_value = [("Generated content", "Used prompt")]
        
        # Configure evaluator mock to return evaluation results
        evaluator_mock.evaluate.return_value = {
            "score": 4.0,
            "feedback": "Good content",
            "improved_output": "Improved content",
            "summary": "Good overall"
        }
        
        # Create test prompts
        prompt1 = MagicMock(spec=Prompt)
        prompt1.template = "Test prompt 1 {input}"
        
        # Initialize the PromptEvaluator with 2 simulations
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations=2)
        
        # Evaluate the prompts
        results = prompt_evaluator.evaluate_prompts([prompt1], "Test input")
        
        # Verify results
        assert len(results) == 1  # One prompt evaluated
        prompt_result, avg_score, simulation_results = results[0]
        
        assert prompt_result == prompt1
        assert avg_score == 4.0  # Average of two 4.0 scores
        assert len(simulation_results) == 2  # Two simulation results
        
        # Verify the generator and evaluator were called correctly
        assert generator_mock.set_prompts.call_count == 1
        generator_mock.set_prompts.assert_called_with([prompt1])
        assert generator_mock.generate.call_count == 2
        assert evaluator_mock.evaluate.call_count == 2
    
    def test_evaluate_multiple_prompts(self):
        """Test evaluation of multiple prompts"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator mock to return generated content
        generator_mock.generate.return_value = [("Generated content", "Used prompt")]
        
        # Configure evaluator mock to return different scores for different evaluations
        evaluation_results = [
            {"score": 3.5, "feedback": "Good content"},
            {"score": 4.0, "feedback": "Better content"},
            {"score": 2.5, "feedback": "Average content"},
            {"score": 4.5, "feedback": "Excellent content"}
        ]
        evaluator_mock.evaluate.side_effect = evaluation_results
        
        # Create test prompts
        prompt1 = MagicMock(spec=Prompt)
        prompt1.template = "Test prompt 1 {input}"
        prompt2 = MagicMock(spec=Prompt)
        prompt2.template = "Test prompt 2 {input}"
        
        # Initialize the PromptEvaluator with 2 simulations
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations=2)
        
        # Evaluate the prompts
        results = prompt_evaluator.evaluate_prompts([prompt1, prompt2], "Test input")
        
        # Verify results
        assert len(results) == 2  # Two prompts evaluated
        
        # First prompt should have average score of (3.5 + 4.0) / 2 = 3.75
        prompt1_result, avg_score1, simulation_results1 = results[0]
        assert prompt1_result == prompt1
        assert avg_score1 == 3.75
        assert len(simulation_results1) == 2
        
        # Second prompt should have average score of (2.5 + 4.5) / 2 = 3.5
        prompt2_result, avg_score2, simulation_results2 = results[1]
        assert prompt2_result == prompt2
        assert avg_score2 == 3.5
        assert len(simulation_results2) == 2
        
        # Verify the generator was called with each prompt
        assert generator_mock.set_prompts.call_count == 2
        generator_mock.set_prompts.assert_any_call([prompt1])
        generator_mock.set_prompts.assert_any_call([prompt2])
    
    def test_empty_generation_result(self):
        """Test handling of empty generation results"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator mock to return empty results
        generator_mock.generate.return_value = []
        
        # Create test prompt
        prompt = MagicMock(spec=Prompt)
        prompt.template = "Test prompt {input}"
        
        # Initialize the PromptEvaluator
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock)
        
        # Evaluate the prompt
        results = prompt_evaluator.evaluate_prompts([prompt], "Test input")
        
        # Verify results
        assert len(results) == 1
        prompt_result, avg_score, simulation_results = results[0]
        
        assert prompt_result == prompt
        assert avg_score == 0  # No scores, so average is 0
        assert len(simulation_results) == 0  # No simulation results
        
        # Verify the evaluator was not called
        evaluator_mock.evaluate.assert_not_called()
    
    def test_varying_scores(self):
        """Test with varying scores in simulations"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator mock to return generated content
        generator_mock.generate.return_value = [("Generated content", "Used prompt")]
        
        # Configure evaluator mock to return different scores for each call
        evaluation_results = [
            {"score": 2.0, "feedback": "Poor content"},
            {"score": 3.0, "feedback": "Average content"},
            {"score": 5.0, "feedback": "Excellent content"}
        ]
        evaluator_mock.evaluate.side_effect = evaluation_results
        
        # Create test prompt
        prompt = MagicMock(spec=Prompt)
        prompt.template = "Test prompt {input}"
        
        # Initialize the PromptEvaluator with 3 simulations
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations=3)
        
        # Evaluate the prompt
        results = prompt_evaluator.evaluate_prompts([prompt], "Test input")
        
        # Verify results
        prompt_result, avg_score, simulation_results = results[0]
        
        # Average score should be (2.0 + 3.0 + 5.0) / 3 = 3.33...
        assert prompt_result == prompt
        assert abs(avg_score - 3.33) < 0.01  # Approximately 3.33
        assert len(simulation_results) == 3
        
        # Verify the evaluator was called the right number of times
        assert evaluator_mock.evaluate.call_count == 3
    
    def test_select_best_prompts(self):
        """Test selecting top performing prompts by quantile (extension method)"""
        def select_best_prompts(prompt_results, quantile=0.5):
            """
            Select the best performing prompts based on a quantile threshold.
            
            Args:
                prompt_results: List of (prompt, score, details) tuples from evaluate_prompts
                quantile: Fraction of prompts to keep (0.5 means top 50%)
                
            Returns:
                List of (prompt, score, details) tuples for the best prompts
            """
            # Sort by score in descending order
            sorted_results = sorted(prompt_results, key=lambda x: x[1], reverse=True)
            
            # Calculate how many prompts to keep
            keep_count = max(1, int(len(sorted_results) * quantile))
            
            # Return the top performers
            return sorted_results[:keep_count]
        
        # Create a list of test prompts with scores
        prompt1 = MagicMock(spec=Prompt)
        prompt1.template = "Prompt 1"
        
        prompt2 = MagicMock(spec=Prompt)
        prompt2.template = "Prompt 2"
        
        prompt3 = MagicMock(spec=Prompt)
        prompt3.template = "Prompt 3"
        
        prompt4 = MagicMock(spec=Prompt)
        prompt4.template = "Prompt 4"
        
        prompt5 = MagicMock(spec=Prompt)
        prompt5.template = "Prompt 5"
        
        # Create prompt results with scores
        prompt_results = [
            (prompt1, 3.0, [{"score": 3.0}]),
            (prompt2, 4.5, [{"score": 4.5}]),
            (prompt3, 2.5, [{"score": 2.5}]),
            (prompt4, 4.0, [{"score": 4.0}]),
            (prompt5, 3.8, [{"score": 3.8}])
        ]
        
        # Select top 40% of prompts
        best_prompts = select_best_prompts(prompt_results, quantile=0.4)
        
        # Should keep 2 out of 5 prompts (40%)
        assert len(best_prompts) == 2
        # Should select prompts with scores 4.5 and 4.0 (the top 2)
        assert best_prompts[0][1] == 4.5
        assert best_prompts[1][1] == 4.0
    
    def test_edge_case_identical_content(self):
        """Test with prompts that generate identical content"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator to always return the same content
        generator_mock.generate.return_value = [("Identical content", "Used prompt")]
        
        # Configure evaluator to always return the same score
        evaluator_mock.evaluate.return_value = {"score": 3.0, "feedback": "Standard feedback"}
        
        # Create test prompts
        prompt1 = MagicMock(spec=Prompt)
        prompt2 = MagicMock(spec=Prompt)
        
        # Initialize the PromptEvaluator
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations=1)
        
        # Evaluate the prompts
        results = prompt_evaluator.evaluate_prompts([prompt1, prompt2], "Test input")
        
        # Both prompts should have the same score
        assert results[0][1] == results[1][1] == 3.0
    
    def test_edge_case_all_equal_scores(self):
        """Test with all prompts scoring equally"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator to return different content but evaluator gives same score
        generator_mock.generate.side_effect = [
            [("Content 1", "Used prompt")],
            [("Content 2", "Used prompt")],
            [("Content 3", "Used prompt")]
        ]
        
        # Configure evaluator to always return the same score
        evaluator_mock.evaluate.return_value = {"score": 4.0, "feedback": "Equal quality"}
        
        # Create test prompts
        prompts = [MagicMock(spec=Prompt) for _ in range(3)]
        
        # Initialize the PromptEvaluator
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations=1)
        
        # Evaluate the prompts
        results = prompt_evaluator.evaluate_prompts(prompts, "Test input")
        
        # All prompts should have the same score
        assert all(result[1] == 4.0 for result in results)
    
    def test_edge_case_minimal_separation(self):
        """Test with minimal separation between prompt scores"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator mock to return generated content
        generator_mock.generate.return_value = [("Generated content", "Used prompt")]
        
        # Configure evaluator to return very close scores
        evaluator_mock.evaluate.side_effect = [
            {"score": 4.001, "feedback": "Slightly better"},
            {"score": 4.000, "feedback": "Good"},
            {"score": 3.999, "feedback": "Almost as good"}
        ]
        
        # Create test prompts
        prompts = [MagicMock(spec=Prompt) for _ in range(3)]
        
        # Initialize the PromptEvaluator
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations=1)
        
        # Evaluate the prompts
        results = prompt_evaluator.evaluate_prompts(prompts, "Test input")
        
        # Verify the scores have minimal separation
        scores = [result[1] for result in results]
        assert max(scores) - min(scores) < 0.01
    
    def test_edge_case_fewer_prompts_than_selection_size(self):
        """Test with fewer prompts than the required selection size"""
        # Additional test to cover the edge case specified in requirements
        def select_best_prompts(prompt_results, quantile=0.5):
            # Same implementation as in test_select_best_prompts
            sorted_results = sorted(prompt_results, key=lambda x: x[1], reverse=True)
            keep_count = max(1, int(len(sorted_results) * quantile))
            return sorted_results[:keep_count]
        
        # Create a single prompt result
        prompt1 = MagicMock(spec=Prompt)
        prompt_results = [(prompt1, 4.0, [{"score": 4.0}])]
        
        # Try to select top 40% when there's only 1 prompt
        best_prompts = select_best_prompts(prompt_results, quantile=0.4)
        
        # Should still keep at least 1 prompt
        assert len(best_prompts) == 1
        assert best_prompts[0][1] == 4.0
    
    def test_evaluate_with_multiple_inputs(self):
        """Test evaluation with multiple input data items"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator mock to return generated content
        generator_mock.generate.return_value = [("Generated content", "Used prompt")]
        
        # Configure evaluator mock to return evaluation results
        # Different scores for different inputs
        evaluator_mock.evaluate.side_effect = [
            {"score": 3.0, "feedback": "Input 1 feedback"},
            {"score": 4.0, "feedback": "Input 2 feedback"},
            {"score": 5.0, "feedback": "Input 3 feedback"},
        ]
        
        # Create test prompt
        prompt = MagicMock(spec=Prompt)
        prompt.template = "Test prompt {input}"
        
        # Create multiple input data items
        input_data = ["Input 1", "Input 2", "Input 3"]
        
        # Initialize the PromptEvaluator with 3 simulations
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, num_simulations=3)
        
        # Evaluate the prompt with multiple inputs
        results = prompt_evaluator.evaluate_prompts([prompt], input_data)
        
        # Verify results
        assert len(results) == 1
        prompt_result, avg_score, simulation_results = results[0]
        
        # Average score should be (3.0 + 4.0 + 5.0) / 3 = 4.0
        assert prompt_result == prompt
        assert avg_score == 4.0
        assert len(simulation_results) == 3
        
        # Verify that generator.generate was called with each input
        assert generator_mock.generate.call_count == 3
        for i, call in enumerate(generator_mock.generate.call_args_list):
            # Assert each call used the expected input
            assert call[0][0] == input_data[i]
    
    def test_select_best_prompts(self):
        """Test selecting the best prompts based on a quantile threshold"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Create test prompt evaluator with a quantile threshold of 0.4
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock, quantile_threshold=0.4)
        
        # Create test prompts
        prompt1 = MagicMock(spec=Prompt)
        prompt2 = MagicMock(spec=Prompt)
        prompt3 = MagicMock(spec=Prompt)
        prompt4 = MagicMock(spec=Prompt)
        prompt5 = MagicMock(spec=Prompt)
        
        # Create prompt evaluation results
        prompt_results = [
            (prompt1, 3.0, [{"score": 3.0}]),
            (prompt2, 4.5, [{"score": 4.5}]),
            (prompt3, 2.5, [{"score": 2.5}]),
            (prompt4, 4.0, [{"score": 4.0}]),
            (prompt5, 3.8, [{"score": 3.8}])
        ]
        
        # Select best prompts with default quantile (0.4)
        best_prompts = prompt_evaluator.select_best_prompts(prompt_results)
        
        # Should keep 2 out of 5 prompts (40%)
        assert len(best_prompts) == 2
        # Should select prompts with scores 4.5 and 4.0 (the top 2)
        assert best_prompts[0][0] == prompt2
        assert best_prompts[0][1] == 4.5
        assert best_prompts[1][0] == prompt4
        assert best_prompts[1][1] == 4.0
        
        # Test with a custom quantile
        best_prompts_custom = prompt_evaluator.select_best_prompts(prompt_results, quantile=0.6)
        
        # Should keep 3 out of 5 prompts (60%)
        assert len(best_prompts_custom) == 3
        # Should select prompts with scores 4.5, 4.0, and 3.8 (the top 3)
        assert best_prompts_custom[0][0] == prompt2
        assert best_prompts_custom[1][0] == prompt4
        assert best_prompts_custom[2][0] == prompt5
    
    def test_evaluate_and_select_best(self):
        """Test evaluating prompts and selecting the best in one operation"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Configure generator mock to return generated content
        generator_mock.generate.return_value = [("Generated content", "Used prompt")]
        
        # Configure evaluator mock to return different scores for different prompts
        evaluator_mock.evaluate.side_effect = [
            {"score": 3.0, "feedback": "Prompt 1 feedback"},
            {"score": 4.5, "feedback": "Prompt 2 feedback"},
            {"score": 2.5, "feedback": "Prompt 3 feedback"},
        ]
        
        # Create test prompts
        prompt1 = MagicMock(spec=Prompt)
        prompt1.template = "Test prompt 1 {input}"
        prompt2 = MagicMock(spec=Prompt)
        prompt2.template = "Test prompt 2 {input}"
        prompt3 = MagicMock(spec=Prompt)
        prompt3.template = "Test prompt 3 {input}"
        
        prompts = [prompt1, prompt2, prompt3]
        
        # Initialize the PromptEvaluator with a quantile threshold of 0.33
        prompt_evaluator = PromptEvaluator(
            generator_mock, evaluator_mock, 
            num_simulations=1, quantile_threshold=0.33
        )
        
        # Evaluate and select the best prompts
        best_prompts = prompt_evaluator.evaluate_and_select_best(
            prompts=prompts,
            input_data="Test input"
        )
        
        # Should keep 1 out of 3 prompts (33%)
        assert len(best_prompts) == 1
        # Should select prompt2 with highest score 4.5
        assert best_prompts[0] == prompt2
        
        # Verify the methods were called with the right parameters
        assert generator_mock.set_prompts.call_count == 3
        assert generator_mock.generate.call_count == 3
        assert evaluator_mock.evaluate.call_count == 3
    
    def test_empty_prompt_list(self):
        """Test behavior with an empty prompt list"""
        # Set up mocks
        generator_mock = MagicMock(spec=GenerativeModel)
        evaluator_mock = MagicMock(spec=AIExpert)
        
        # Initialize the PromptEvaluator
        prompt_evaluator = PromptEvaluator(generator_mock, evaluator_mock)
        
        # Test evaluate_prompts with empty list
        results = prompt_evaluator.evaluate_prompts([], "Test input")
        assert results == []
        
        # Test select_best_prompts with empty list
        best_prompts = prompt_evaluator.select_best_prompts([])
        assert best_prompts == []
        
        # Test evaluate_and_select_best with empty list
        best_prompts = prompt_evaluator.evaluate_and_select_best([], "Test input")
        assert best_prompts == []
        
        # Verify no calls were made to the mocks
        generator_mock.set_prompts.assert_not_called()
        generator_mock.generate.assert_not_called()
        evaluator_mock.evaluate.assert_not_called() 