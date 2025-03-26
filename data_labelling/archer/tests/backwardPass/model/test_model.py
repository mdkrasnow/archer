"""
Tests for the Model class in data-labelling/archer/backwardPass/model.py.
"""

import pytest
import sys
import os
import tempfile
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backwardPass.model import Model
from backwardPass.promptOptimizer import PromptOptimizer
from helpers.prompt import Prompt


class TestModel:
    """Test suite for the Model class."""

    def test_model_initialization(self):
        """Test that the Model initializes with correct parameters."""
        # Basic initialization
        model = Model(name="test_model")
        assert model.name == "test_model"
        assert model.prompts == {}
        assert model.functions == {}
        assert model.model_type == "generator"
        assert model.adalflow_enabled is False
        assert model.version == "1.0.0"
        assert isinstance(model.metadata, dict)
        assert model.performance_history == []
        
        # Initialize with prompts
        test_prompt = Prompt(content="Test prompt content")
        model = Model(
            name="test_model_with_prompts",
            prompts={"main_prompt": test_prompt}
        )
        assert "main_prompt" in model.prompts
        assert model.prompts["main_prompt"] is test_prompt
        
        # Initialize with AdaLflow enabled
        model = Model(
            name="test_model_with_adalflow",
            adalflow_enabled=True
        )
        assert model.adalflow_enabled is True
        assert model.adalflow_params == {}  # No prompts yet, so empty adalflow_params
    
    def test_add_and_get_prompt(self):
        """Test adding and retrieving prompts from the model."""
        model = Model(name="test_model")
        test_prompt = Prompt(content="Test prompt content")
        
        # Add a prompt
        model.add_prompt("main_prompt", test_prompt)
        assert "main_prompt" in model.prompts
        assert model.prompts["main_prompt"] is test_prompt
        
        # Get the prompt
        retrieved_prompt = model.get_prompt("main_prompt")
        assert retrieved_prompt is test_prompt
        
        # Get a non-existent prompt
        assert model.get_prompt("non_existent") is None
        
        # Test with AdaLflow enabled
        model = Model(name="test_model", adalflow_enabled=True)
        model.add_prompt("main_prompt", test_prompt)
        assert "main_prompt" in model.adalflow_params
        assert model.adalflow_params["main_prompt"].data == test_prompt.content

    def test_remove_prompt(self):
        """Test removing prompts from the model."""
        model = Model(name="test_model")
        test_prompt = Prompt(content="Test prompt content")
        
        # Add and then remove a prompt
        model.add_prompt("main_prompt", test_prompt)
        assert "main_prompt" in model.prompts
        
        result = model.remove_prompt("main_prompt")
        assert result is True
        assert "main_prompt" not in model.prompts
        
        # Try to remove a non-existent prompt
        result = model.remove_prompt("non_existent")
        assert result is False
        
        # Test with AdaLflow enabled
        model = Model(name="test_model", adalflow_enabled=True)
        model.add_prompt("main_prompt", test_prompt)
        assert "main_prompt" in model.adalflow_params
        
        model.remove_prompt("main_prompt")
        assert "main_prompt" not in model.adalflow_params

    def test_add_and_get_function(self):
        """Test adding and retrieving functions from the model."""
        model = Model(name="test_model")
        
        # Define a test function
        def test_function(input_data, model):
            return f"Processed: {input_data}"
        
        # Add the function
        model.add_function("process", test_function)
        assert "process" in model.functions
        assert model.functions["process"] is test_function
        
        # Get the function
        retrieved_function = model.get_function("process")
        assert retrieved_function is test_function
        
        # Get a non-existent function
        assert model.get_function("non_existent") is None

    def test_remove_function(self):
        """Test removing functions from the model."""
        model = Model(name="test_model")
        
        # Define a test function
        def test_function(input_data, model):
            return f"Processed: {input_data}"
        
        # Add and then remove a function
        model.add_function("process", test_function)
        assert "process" in model.functions
        
        result = model.remove_function("process")
        assert result is True
        assert "process" not in model.functions
        
        # Try to remove a non-existent function
        result = model.remove_function("non_existent")
        assert result is False

    def test_update_prompt(self):
        """Test updating a prompt in the model."""
        model = Model(name="test_model")
        test_prompt = Prompt(content="Original content")
        
        # Add the prompt
        model.add_prompt("main_prompt", test_prompt)
        
        # Update the prompt
        result = model.update_prompt("main_prompt", "New content", 4.5, "Good job!")
        assert result is True
        
        # Verify the changes
        prompt = model.get_prompt("main_prompt")
        assert prompt.content == "New content"
        assert prompt.score == 4.5
        assert prompt.feedback == "Good job!"
        assert prompt.generation == 1  # Should increment
        assert len(prompt.history) == 1  # Should have one history entry
        
        # Try to update a non-existent prompt
        result = model.update_prompt("non_existent", "New content")
        assert result is False
        
        # Test with AdaLflow enabled
        model = Model(name="test_model", adalflow_enabled=True)
        model.add_prompt("main_prompt", test_prompt)
        
        model.update_prompt("main_prompt", "Updated content")
        assert model.adalflow_params["main_prompt"].data == "Updated content"

    @patch('backwardPass.promptOptimizer.PromptOptimizer.optimize_prompt')
    def test_optimize_prompt(self, mock_optimize):
        """Test optimizing a prompt in the model."""
        # Setup mock
        mock_optimize.return_value = "Improved content"
        
        # Setup model and optimizer
        model = Model(name="test_model")
        test_prompt = Prompt(content="Original content")
        model.add_prompt("main_prompt", test_prompt)
        
        optimizer = PromptOptimizer(model_name="gemini-2.0-flash")
        
        # Optimize the prompt
        result = model.optimize_prompt("main_prompt", optimizer, "Need improvements", 3.0)
        assert result is True
        
        # Verify the optimization was called
        mock_optimize.assert_called_once_with(test_prompt, "Need improvements", 3.0)
        
        # Verify the prompt was updated
        prompt = model.get_prompt("main_prompt")
        assert prompt.content == "Improved content"
        assert prompt.score == 3.0
        assert prompt.feedback == "Need improvements"
        
        # Try to optimize a non-existent prompt
        result = model.optimize_prompt("non_existent", optimizer, "Feedback", 2.0)
        assert result is False

    def test_evaluate(self):
        """Test evaluating the model."""
        # Setup model with functions
        model = Model(name="test_model")
        
        # Create mock functions
        def function1(input_data, model):
            return f"Function 1 output from {input_data}"
        
        def function2(input_data, model):
            return f"Function 2 output from {input_data}"
        
        model.add_function("func1", function1)
        model.add_function("func2", function2)
        
        # Create a mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = [
            {"score": 4.0, "feedback": "Good job on function 1!"},
            {"score": 3.0, "feedback": "Function 2 needs improvement."}
        ]
        
        # Evaluate the model
        results = model.evaluate(mock_evaluator, "test input")
        
        # Verify evaluator was called for each function
        assert mock_evaluator.evaluate.call_count == 2
        
        # Verify results contain function outputs and evaluations
        assert "func1" in results
        assert "func2" in results
        assert results["func1"]["content"] == "Function 1 output from test input"
        assert results["func2"]["content"] == "Function 2 output from test input"
        assert results["func1"]["evaluation"]["score"] == 4.0
        assert results["func2"]["evaluation"]["score"] == 3.0
        
        # Verify overall score is calculated
        assert "overall_score" in results
        assert results["overall_score"] == 3.5  # Average of 4.0 and 3.0
        
        # Verify performance history is updated
        assert len(model.performance_history) == 1
        assert model.performance_history[0]["overall_score"] == 3.5

    def test_to_dict_and_from_dict(self):
        """Test serializing and deserializing the model."""
        # Setup original model
        model = Model(
            name="test_model", 
            model_type="custom_type",
            version="2.0.0", 
            metadata={"key": "value"}
        )
        
        # Add prompts and functions
        model.add_prompt("prompt1", Prompt(content="Prompt 1 content"))
        model.add_prompt("prompt2", Prompt(content="Prompt 2 content", score=4.5, feedback_or_generation="Good prompt"))
        
        def function1(input_data, model):
            return f"Function 1 output from {input_data}"
        
        model.add_function("func1", function1)
        
        # Convert to dictionary
        model_dict = model.to_dict()
        
        # Verify dictionary contents
        assert model_dict["name"] == "test_model"
        assert model_dict["model_type"] == "custom_type"
        assert model_dict["version"] == "2.0.0"
        assert model_dict["metadata"] == {"key": "value"}
        assert "prompt1" in model_dict["prompts"]
        assert "prompt2" in model_dict["prompts"]
        assert model_dict["prompts"]["prompt1"]["content"] == "Prompt 1 content"
        assert model_dict["prompts"]["prompt2"]["content"] == "Prompt 2 content"
        assert model_dict["prompts"]["prompt2"]["score"] == 4.5
        assert model_dict["function_ids"] == ["func1"]
        
        # Create a new model from the dictionary
        functions = {"func1": function1}
        new_model = Model.from_dict(model_dict, functions)
        
        # Verify the new model matches the original
        assert new_model.name == model.name
        assert new_model.model_type == model.model_type
        assert new_model.version == model.version
        assert new_model.metadata == model.metadata
        assert "prompt1" in new_model.prompts
        assert "prompt2" in new_model.prompts
        assert new_model.prompts["prompt1"].content == "Prompt 1 content"
        assert new_model.prompts["prompt2"].content == "Prompt 2 content"
        assert new_model.prompts["prompt2"].score == 4.5
        assert "func1" in new_model.functions
        assert new_model.functions["func1"] is function1

    def test_save_to_file_and_load_from_file(self):
        """Test saving and loading the model to/from a file."""
        # Setup model
        model = Model(name="test_model")
        model.add_prompt("prompt1", Prompt(content="Prompt 1 content"))
        
        def function1(input_data, model):
            return f"Function 1 output from {input_data}"
        
        model.add_function("func1", function1)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            model.save_to_file(filepath)
            
            # Load the model from the file
            functions = {"func1": function1}
            loaded_model = Model.load_from_file(filepath, functions)
            
            # Verify the loaded model matches the original
            assert loaded_model.name == model.name
            assert "prompt1" in loaded_model.prompts
            assert loaded_model.prompts["prompt1"].content == "Prompt 1 content"
            assert "func1" in loaded_model.functions
            assert loaded_model.functions["func1"] is function1
        finally:
            # Clean up the temporary file
            os.unlink(filepath)

    def test_clone(self):
        """Test cloning a model."""
        # Setup original model
        model = Model(name="test_model")
        model.add_prompt("prompt1", Prompt(content="Prompt 1 content"))
        
        def function1(input_data, model):
            return f"Function 1 output from {input_data}"
        
        model.add_function("func1", function1)
        
        # Clone the model
        cloned_model = model.clone()
        
        # Verify the clone matches the original
        assert cloned_model.name == model.name
        assert "prompt1" in cloned_model.prompts
        assert cloned_model.prompts["prompt1"].content == "Prompt 1 content"
        assert "func1" in cloned_model.functions
        assert cloned_model.functions["func1"] is model.functions["func1"]
        
        # Verify that changes to the clone don't affect the original
        cloned_model.prompts["prompt1"].update(new_content="Modified content")
        assert model.prompts["prompt1"].content == "Prompt 1 content"
        assert cloned_model.prompts["prompt1"].content == "Modified content"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 