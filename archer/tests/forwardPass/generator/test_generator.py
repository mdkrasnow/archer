import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from typing import List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from forwardPass.generator import GenerativeModel, default_llm_call
from helpers.prompt import Prompt

class TestGenerativeModel:
    def test_generative_model_initialization(self):
        """Test that a GenerativeModel initializes with correct default values"""
        model_name = "test-model"
        model = GenerativeModel(model_name)
        
        assert model.model_name == model_name
        assert model.temperature == 0.7  # Default value
        assert model.top_p == 0.9  # Default value
        assert model.active_prompts == []
        assert hasattr(model, "generation_func")
        assert hasattr(model, "llm_call")
    
    def test_generative_model_initialization_with_custom_values(self):
        """Test that a GenerativeModel initializes with custom values"""
        model_name = "test-model"
        temperature = 0.5
        top_p = 0.8
        
        model = GenerativeModel(model_name, temperature, top_p)
        
        assert model.model_name == model_name
        assert model.temperature == temperature
        assert model.top_p == top_p
    
    def test_set_prompts(self):
        """Test that prompts can be set and retrieved correctly"""
        model = GenerativeModel("test-model")
        prompts = [Prompt("Prompt 1"), Prompt("Prompt 2")]
        
        model.set_prompts(prompts)
        
        assert model.active_prompts == prompts
        assert len(model.active_prompts) == 2
    
    def test_generate_with_custom_generation_func(self):
        """Test generation with a custom generation function"""
        # Create a mock generation function
        mock_gen_func = MagicMock(return_value="Custom generated content")
        
        model = GenerativeModel("test-model", generation_func=mock_gen_func)
        prompts = [Prompt("Test prompt")]
        model.set_prompts(prompts)
        
        results = model.generate("Test input")
        
        # Verify the mock was called with the correct arguments
        mock_gen_func.assert_called_once_with("Test prompt", "Test input")
        
        # Check the results
        assert len(results) == 1
        assert results[0][0] == "Custom generated content"
        assert results[0][1] == prompts[0]
    
    def test_default_llm_call(self):
        """Test the default LLM call functionality using a mock"""
        # Create a custom mock LLM call function
        mock_llm_call = MagicMock(return_value={
            "choices": [
                {"message": {"content": "Generated content"}}
            ]
        })
        
        # Create model with custom llm_call
        model = GenerativeModel("test-model", llm_call=mock_llm_call)
        result = model._call_llm("Test prompt", "Test input")
        
        # Verify the mock was called with the expected arguments
        mock_llm_call.assert_called_once_with(
            messages=[
                {"role": "system", "content": "Test prompt"},
                {"role": "user", "content": "Test input"}
            ],
            model="test-model",
            temperature=0.7
        )
        
        assert result == "Generated content"
    
    def test_generate_multiple_prompts(self):
        """Test generation with multiple prompts using a mock"""
        # Create a mock LLM call function with multiple return values
        mock_llm_call = MagicMock()
        mock_llm_call.side_effect = [
            {"choices": [{"message": {"content": "Generated content 1"}}]},
            {"choices": [{"message": {"content": "Generated content 2"}}]}
        ]
        
        model = GenerativeModel("test-model", llm_call=mock_llm_call)
        prompts = [Prompt("Prompt 1"), Prompt("Prompt 2")]
        model.set_prompts(prompts)
        
        results = model.generate("Test input")
        
        # Check we got the expected number of results
        assert len(results) == 2
        
        # Check each result
        assert results[0][0] == "Generated content 1"
        assert results[0][1] == prompts[0]
        assert results[1][0] == "Generated content 2"
        assert results[1][1] == prompts[1]
        
        # Verify the mock was called with different prompts
        assert mock_llm_call.call_count == 2
    
    def test_llm_call_error_handling(self):
        """Test handling of errors in LLM responses using a mock"""
        # Create a mock that returns a response without choices
        mock_llm_call = MagicMock(return_value={})
        
        model = GenerativeModel("test-model", llm_call=mock_llm_call)
        result = model._call_llm("Test prompt", "Test input")
        
        assert result == "Error: No response generated"
    
    def test_generate_with_empty_prompts(self):
        """Test generation behavior with no active prompts"""
        model = GenerativeModel("test-model")
        # Don't set any prompts
        
        results = model.generate("Test input")
        
        assert results == []
    
    def test_generate_with_empty_input(self):
        """Test generation with empty input data"""
        mock_gen_func = MagicMock(return_value="Generated from empty input")
        
        model = GenerativeModel("test-model", generation_func=mock_gen_func)
        prompts = [Prompt("Test prompt")]
        model.set_prompts(prompts)
        
        results = model.generate("")
        
        # Verify the mock was called with empty input
        mock_gen_func.assert_called_once_with("Test prompt", "")
        
        assert results[0][0] == "Generated from empty input" 