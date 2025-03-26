# test_promptOptimizer.py
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, create_autospec
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Mock adalflow modules
mock_parameter = MagicMock()
mock_parameter_type = create_autospec(type)
mock_parameter_type.PROMPT = "PROMPT"

# Mock other AdaLFlow components
mock_tgd_optimizer = MagicMock()
mock_generator = MagicMock()
mock_openai_client = MagicMock()

# Create proper mocks that look like types for isinstance checks
sys.modules['adalflow'] = MagicMock()
sys.modules['adalflow.optim'] = MagicMock()
sys.modules['adalflow.optim.parameter'] = MagicMock()
sys.modules['adalflow.optim.parameter'].Parameter = mock_parameter_type
sys.modules['adalflow.optim.parameter'].ParameterType = mock_parameter_type
sys.modules['adalflow.optim.text_grad'] = MagicMock()
sys.modules['adalflow.optim.text_grad.tgd_optimizer'] = MagicMock()
sys.modules['adalflow.optim.text_grad.tgd_optimizer'].TGDOptimizer = mock_tgd_optimizer
sys.modules['adalflow.core'] = MagicMock()
sys.modules['adalflow.core'].Generator = mock_generator
sys.modules['adalflow.core'].Component = MagicMock()
sys.modules['adalflow.core.component'] = MagicMock()
sys.modules['adalflow.core.model_client'] = MagicMock()
sys.modules['adalflow.components'] = MagicMock()
sys.modules['adalflow.components.model_client'] = MagicMock()
sys.modules['adalflow.components.model_client.openai_client'] = MagicMock()
sys.modules['adalflow.components.model_client.openai_client'].OpenAIClient = mock_openai_client

# Import after setting up mocks
from backwardPass.promptOptimizer import PromptOptimizer
from backwardPass.model import Model
from helpers.prompt import Prompt

class TestPromptOptimizer:
    @patch('backwardPass.promptOptimizer.OpenAIClient')
    @patch('backwardPass.promptOptimizer.Generator')
    @patch('backwardPass.promptOptimizer.TGDOptimizer')
    def test_optimizer_initialization(self, mock_tgd, mock_gen, mock_client):
        """Test that PromptOptimizer initializes with correct parameters"""
        mock_gen.return_value = MagicMock()
        mock_tgd.return_value = MagicMock()
        mock_client.return_value = MagicMock()
        
        model_name = "gemini-2.0-flash"
        variation_traits = ["clarity", "specificity"]
        
        optimizer = PromptOptimizer(
            model_name=model_name, 
            temperature=0.8, 
            adalflow_enabled=True,
            max_trials=8,
            top_k=5,
            openrouter_api_key="test_key_123",
            variation_traits=variation_traits
        )
        
        assert optimizer.model_name == model_name
        assert optimizer.temperature == 0.8
        assert optimizer.max_trials == 8
        assert optimizer.top_k == 5
        assert optimizer.openrouter_api_key == "test_key_123"
        assert optimizer.variation_traits == variation_traits
        assert hasattr(optimizer, "llm_call")
    
    def test_gradient_magnitude_calculation(self):
        """Test the gradient magnitude calculation based on scores"""
        optimizer = PromptOptimizer("gemini-2.0-flash")
        
        # Low score should result in high magnitude
        low_mag = optimizer._calculate_gradient_magnitude(1.0, 5.0)
        # Medium score should result in medium magnitude
        med_mag = optimizer._calculate_gradient_magnitude(3.0, 5.0)
        # High score should result in low magnitude
        high_mag = optimizer._calculate_gradient_magnitude(4.5, 5.0)
        
        assert low_mag > med_mag > high_mag
        assert low_mag > 0.7  # Low scores should result in high magnitudes
        assert high_mag < 0.3  # High scores should result in low magnitudes
    
    @patch('backwardPass.promptOptimizer.llm_call')
    def test_prompt_optimization(self, mock_llm_call):
        """Test optimization of a prompt based on feedback and score"""
        # Setup mock llm_call to return a response with content
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Improved prompt that addresses feedback"
                    }
                }
            ]
        }
        mock_llm_call.return_value = mock_response
        
        # Create optimizer and test prompt
        optimizer = PromptOptimizer("gemini-2.0-flash")
        original_prompt = Prompt(
            content="Create a compelling sales email",
        )
        feedback = "Needs more personalization and clearer call to action"
        score = 3.5
        
        # Call the optimize_prompt method
        improved_prompt = optimizer.optimize_prompt(original_prompt, feedback, score)
        
        # Verify llm_call was called correctly
        mock_llm_call.assert_called_once()
        
        # Verify that the prompt includes the score guidance based on the score
        call_args = mock_llm_call.call_args[1]
        prompt_content = call_args['messages'][0]['content']
        assert "Guidance:" in prompt_content
        assert "needs moderate improvement" in prompt_content  # For score 3.5
        
        # Verify that temperature was adjusted based on gradient magnitude
        temp = call_args['temperature']
        assert temp != optimizer.temperature
        
        # Verify result
        assert improved_prompt == "Improved prompt that addresses feedback"
    
    @patch('backwardPass.promptOptimizer.llm_call')
    def test_optimization_with_low_score(self, mock_llm_call):
        """Test optimization with a low score prompt that needs significant improvement"""
        # Setup mock llm_call
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Completely reworked prompt addressing major issues"
                    }
                }
            ]
        }
        mock_llm_call.return_value = mock_response
        
        optimizer = PromptOptimizer("gemini-2.0-flash")
        original_prompt = Prompt(content="Brief email about sale")
        feedback = "Too generic, lacks specific details, no clear audience or benefit"
        score = 1.5  # Very low score
        
        improved_prompt = optimizer.optimize_prompt(original_prompt, feedback, score)
        
        # Verify the optimization was performed
        mock_llm_call.assert_called_once()
        
        # Check that low score prompts get the significant improvement guidance
        call_args = mock_llm_call.call_args[1]
        prompt_content = call_args['messages'][0]['content']
        assert "significant improvement" in prompt_content
        
        # Verify higher temperature for low scores (more aggressive changes)
        temp = call_args['temperature']
        assert temp > optimizer.temperature
        
        # Verify result
        assert improved_prompt == "Completely reworked prompt addressing major issues"
    
    @patch('backwardPass.promptOptimizer.llm_call')
    def test_optimization_with_high_score(self, mock_llm_call):
        """Test optimization with a high score prompt that needs minimal changes"""
        # Setup mock llm_call
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Slightly refined version of an already excellent prompt"
                    }
                }
            ]
        }
        mock_llm_call.return_value = mock_response
        
        optimizer = PromptOptimizer("gemini-2.0-flash")
        original_prompt = Prompt(content="Create a personalized email highlighting key benefits of our premium service")
        feedback = "Already strong, just needs minor refinement in tone"
        score = 4.8  # Very high score
        
        improved_prompt = optimizer.optimize_prompt(original_prompt, feedback, score)
        
        # Verify the optimization was performed
        mock_llm_call.assert_called_once()
        
        # Check high score prompts get the minor refinement guidance
        call_args = mock_llm_call.call_args[1]
        prompt_content = call_args['messages'][0]['content']
        assert "already quite good" in prompt_content
        
        # Verify lower temperature for high scores (more conservative changes)
        temp = call_args['temperature']
        assert abs(temp - optimizer.temperature) < 0.2  # Minimal temperature adjustment
        
        # Verify result
        assert improved_prompt == "Slightly refined version of an already excellent prompt"
    
    @patch('backwardPass.promptOptimizer.llm_call', side_effect=Exception("LLM error"))
    def test_error_handling(self, mock_llm_call):
        """Test handling of LLM call integration failures"""
        optimizer = PromptOptimizer("gemini-2.0-flash")
        original_prompt = Prompt(content="Original prompt content")
        feedback = "Some feedback"
        score = 3.0
        
        # Even if the LLM call raises an exception, we should get back the original prompt
        improved_prompt = optimizer.optimize_prompt(original_prompt, feedback, score)
        
        # Verify result is the original prompt content (error handling)
        assert improved_prompt == original_prompt.content
        
        mock_llm_call.assert_called_once()

    @patch('backwardPass.promptOptimizer.llm_call')
    def test_generate_prompt_variants(self, mock_llm_call):
        """Test generation of prompt variants with natural variation"""
        # Setup mock responses for multiple calls
        mock_responses = [
            {"choices": [{"message": {"content": "Variant 1 of the prompt"}}]},
            {"choices": [{"message": {"content": "Variant 2 of the prompt"}}]},
            {"choices": [{"message": {"content": "Variant 3 of the prompt"}}]},
        ]
        mock_llm_call.side_effect = mock_responses
        
        optimizer = PromptOptimizer(
            "gemini-2.0-flash", 
            variation_traits=["clarity", "specificity"]
        )
        base_prompt = Prompt(content="Create a sales email", score=4.0)
        
        # Generate variants
        variants = optimizer.generate_prompt_variants(
            [base_prompt], 
            num_variants=3
        )
        
        # Verify correct number of variants generated
        assert len(variants) == 3
        
        # Verify different content for each variant
        variant_contents = [v.content for v in variants]
        assert len(set(variant_contents)) == 3
        
        # Verify each has incremented generation number
        for v in variants:
            assert v.generation == base_prompt.generation + 1
        
        # Verify the variation traits were included in the LLM prompts
        for call_args in mock_llm_call.call_args_list:
            prompt_content = call_args[1]['messages'][0]['content']
            assert "Create a variation" in prompt_content
            assert "clarity" in prompt_content
            assert "specificity" in prompt_content

    # Skip this test as we can't properly mock the adalflow components
    def test_wrap_prompts_as_params(self):
        """Test that prompts can be wrapped as AdaLFlow parameters"""
        pytest.skip("This test requires a real AdaLFlow installation")

    @patch('backwardPass.promptOptimizer.llm_call')
    @patch('backwardPass.promptOptimizer.ADALFLOW_AVAILABLE', False)
    def test_optimize_with_adalflow_disabled(self, mock_llm_call):
        """Test optimizing prompts with AdaLFlow disabled - should fallback to standard approach"""
        # Setup mock for optimize_prompt to avoid deep LLM calls
        mock_responses = [
            {"choices": [{"message": {"content": "Improved prompt 1"}}]},
            {"choices": [{"message": {"content": "Improved prompt 2"}}]},
            {"choices": [{"message": {"content": "Variant 1"}}]},
            {"choices": [{"message": {"content": "Variant 2"}}]},
        ]
        mock_llm_call.side_effect = mock_responses
        
        optimizer = PromptOptimizer("gemini-2.0-flash", adalflow_enabled=True)
        prompts = [
            Prompt(content="Prompt 1", score=3.5, generation=1),
            Prompt(content="Prompt 2", score=4.0, generation=1)
        ]
        feedback_map = {"0": "Feedback 1", "1": "Feedback 2"}
        score_map = {"0": 3.5, "1": 4.0}
        
        # Call optimize method
        new_prompts = optimizer.optimize(prompts, feedback_map, score_map)
        
        # Verify correct number of prompts returned (2 original improved + variants)
        assert len(new_prompts) > 2
        
        # Verify the first two prompts have been improved
        assert new_prompts[0].content == "Improved prompt 1"
        assert new_prompts[1].content == "Improved prompt 2"
        
        # Verify additional variants were generated
        assert any(p.content == "Variant 1" for p in new_prompts)
        assert any(p.content == "Variant 2" for p in new_prompts)
    
    @patch('backwardPass.promptOptimizer.OpenAIClient')
    @patch('backwardPass.promptOptimizer.Generator')
    @patch('backwardPass.promptOptimizer.TGDOptimizer')
    def test_optimize_model_prompts(self, mock_tgd, mock_gen, mock_client):
        """Test optimizing prompts in a Model instance"""
        # Setup mocks
        mock_gen.return_value = MagicMock()
        mock_tgd.return_value = MagicMock()
        mock_client.return_value = MagicMock()
        
        # Reset mocks
        mock_parameter.reset_mock()
        mock_tgd_optimizer.reset_mock()

        # Create mock parameter instances for the model
        model_param1 = MagicMock()
        model_param1.data = "Updated model prompt 1"
        model_param2 = MagicMock()
        model_param2.data = "Updated model prompt 2"

        # Create a model with mocked params
        model = MagicMock()
        model.adalflow_enabled = True
        model.adalflow_params = {
            "prompt1": model_param1,
            "prompt2": model_param2
        }
        model.prompts = {
            "prompt1": Prompt(content="Original prompt 1"),
            "prompt2": Prompt(content="Original prompt 2")
        }

        # Create the optimizer with AdaLFlow enabled
        optimizer = PromptOptimizer("gemini-2.0-flash", adalflow_enabled=True)
        
        # Create feedback and score maps
        feedback_map = {
            "prompt1": "Feedback 1",
            "prompt2": "Feedback 2"
        }
        score_map = {
            "prompt1": 3.5,
            "prompt2": 4.2
        }
        
        # Call optimize_model
        result = optimizer.optimize_model(model, feedback_map, score_map)
        
        # Verify the model's update_prompt method was called
        assert model.update_prompt.call_count == 2
        
        # Verify gradients were added to parameters with magnitude information
        assert model_param1.add_gradient.called
        gradient_data = model_param1.add_gradient.call_args[0][0]
        assert "magnitude" in gradient_data
        assert "score" in gradient_data
        assert "feedback" in gradient_data
        assert "variation_traits" in gradient_data
        
        assert result is True
    
    @patch('backwardPass.promptOptimizer.llm_call')
    @patch('backwardPass.promptOptimizer.OpenAIClient')
    @patch('backwardPass.promptOptimizer.Generator')
    @patch('backwardPass.promptOptimizer.TGDOptimizer')
    def test_optimize_model_with_evaluation(self, mock_tgd, mock_gen, mock_client, mock_llm_call):
        """Test optimize_model_with_evaluation method"""
        # Setup mocks
        mock_gen.return_value = MagicMock()
        mock_tgd.return_value = MagicMock()
        mock_client.return_value = MagicMock()
        
        # Mock responses for variant generation
        mock_responses = [
            {"choices": [{"message": {"content": "Variant 1"}}]},
            {"choices": [{"message": {"content": "Variant 2"}}]},
            {"choices": [{"message": {"content": "Variant 3"}}]},
        ]
        mock_llm_call.side_effect = mock_responses
        
        # Setup model
        model = MagicMock()
        model.adalflow_enabled = True
        model.prompts = {
            "prompt1": Prompt(content="Original prompt 1"),
            "prompt2": Prompt(content="Original prompt 2")
        }
        
        # Mock prompt evaluator
        prompt_evaluator = MagicMock()
        best_prompts = [Prompt(content="Best prompt", score=4.8)]
        prompt_evaluator.evaluate_and_select_best.return_value = best_prompts
        
        # Create the optimizer
        optimizer = PromptOptimizer(
            "gemini-2.0-flash", 
            adalflow_enabled=True,
            variation_traits=["clarity"]
        )
        
        # Mock optimize_model to return True
        optimizer.optimize_model = MagicMock(return_value=True)
        
        # Test the method
        feedback_map = {"prompt1": "Feedback 1", "prompt2": "Feedback 2"}
        score_map = {"prompt1": 3.5, "prompt2": 4.2}
        input_data = "Test input data"
        
        result = optimizer.optimize_model_with_evaluation(
            model, feedback_map, score_map, input_data, prompt_evaluator
        )
        
        # Verify optimize_model was called
        optimizer.optimize_model.assert_called_once()
        
        # Verify evaluate_and_select_best was called
        prompt_evaluator.evaluate_and_select_best.assert_called_once()
        
        # Verify the result is the best prompts from evaluator
        assert result == best_prompts

    @patch('backwardPass.promptOptimizer.llm_call')
    @patch('backwardPass.promptOptimizer.OpenAIClient')
    @patch('backwardPass.promptOptimizer.Generator')
    @patch('backwardPass.promptOptimizer.TGDOptimizer')
    def test_optimize_model_with_evaluation_no_evaluator(self, mock_tgd, mock_gen, mock_client, mock_llm_call):
        """Test optimize_model_with_evaluation method when no evaluator is provided"""
        # Setup mocks
        mock_gen.return_value = MagicMock()
        mock_tgd.return_value = MagicMock()
        mock_client.return_value = MagicMock()
        
        # Mock responses for variant generation
        mock_responses = [
            {"choices": [{"message": {"content": "Variant 1"}}]},
            {"choices": [{"message": {"content": "Variant 2"}}]},
        ]
        mock_llm_call.side_effect = mock_responses
        
        # Setup model
        model = MagicMock()
        model.adalflow_enabled = True
        model.prompts = {
            "prompt1": Prompt(content="Original prompt 1"),
            "prompt2": Prompt(content="Original prompt 2")
        }
        
        # Create the optimizer
        optimizer = PromptOptimizer("gemini-2.0-flash", adalflow_enabled=True)
        
        # Mock optimize_model to return True
        optimizer.optimize_model = MagicMock(return_value=True)
        
        # Test the method without an evaluator
        feedback_map = {"prompt1": "Feedback 1", "prompt2": "Feedback 2"}
        score_map = {"prompt1": 3.5, "prompt2": 4.2}
        input_data = "Test input data"
        
        result = optimizer.optimize_model_with_evaluation(
            model, feedback_map, score_map, input_data, None
        )
        
        # Verify optimize_model was called
        optimizer.optimize_model.assert_called_once()
        
        # Verify the result includes both original prompts and variants
        assert len(result) >= 2  # At least the original prompts