"""
Tests for the DanielsonModel class in data-labelling/archer/backwardPass/danielson_model.py.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Mock the eval.danielson module since it's not directly accessible in tests
sys.modules['eval'] = MagicMock()
sys.modules['eval.danielson'] = MagicMock()
sys.modules['eval.danielson'].generate_ai_content = MagicMock()
sys.modules['eval.danielson'].normalize_score_integer = MagicMock(return_value=3)

from backwardPass.danielson_model import DanielsonModel


class TestDanielsonModel:
    """Test suite for the DanielsonModel class."""
    
    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        """Reset mocks before each test."""
        sys.modules['eval.danielson'].generate_ai_content.reset_mock()
        sys.modules['eval.danielson'].generate_ai_content.side_effect = None
        sys.modules['eval.danielson'].normalize_score_integer.reset_mock()
        sys.modules['eval.danielson'].normalize_score_integer.return_value = 3
        yield

    def test_danielson_model_initialization(self):
        """Test that the DanielsonModel initializes with correct parameters."""
        # Basic initialization
        model = DanielsonModel()
        assert model.name == "danielson"
        assert model.model_type == "evaluator"
        assert model.adalflow_enabled is False
        assert model.version == "1.0.0"
        assert isinstance(model.metadata, dict)
        
        # Verify Danielson-specific prompts were created
        assert "context_analysis" in model.prompts
        assert "component_evaluation_base" in model.prompts
        assert "restructure_feedback" in model.prompts
        assert "component_instruction_1a" in model.prompts
        assert "component_instruction_3e" in model.prompts
        
        # Verify Danielson functions were registered
        assert "analyze_context" in model.functions
        assert "generate_component_evaluation" in model.functions
        assert "restructure_feedback" in model.functions
        assert "generate_single_evaluation" in model.functions
        
        # Initialize with custom parameters
        model = DanielsonModel(
            name="custom_danielson",
            adalflow_enabled=True,
            version="2.0.0",
            metadata={"creator": "Test User"}
        )
        assert model.name == "custom_danielson"
        assert model.adalflow_enabled is True
        assert model.version == "2.0.0"
        assert model.metadata == {"creator": "Test User"}
        
        # Verify AdaLflow parameters were created
        assert len(model.adalflow_params) > 0
        assert "context_analysis" in model.adalflow_params

    def test_analyze_danielson_context(self):
        """Test analyzing Danielson context."""
        # Setup mock
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Mock analysis content"
        mock_response.parts = [mock_part]
        sys.modules['eval.danielson'].generate_ai_content.return_value = mock_response
        
        # Create model
        model = DanielsonModel()
        
        # Call analyze_danielson_context
        result = model.analyze_danielson_context("Sample observation text")
        
        # Verify the function was called
        sys.modules['eval.danielson'].generate_ai_content.assert_called_once()
        
        # Verify the result
        assert result["analysis"] == "Mock analysis content"
        assert result["error"] is None
        
        # Test error handling
        sys.modules['eval.danielson'].generate_ai_content.side_effect = Exception("Test error")
        result = model.analyze_danielson_context("Sample observation text")
        assert result["analysis"] == ""
        assert result["error"] == "Test error"

    def test_generate_component_evaluation(self):
        """Test generating component evaluation."""
        # Setup mocks
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = '{"score": 3, "summary": "Sample evaluation summary"}'
        mock_response.parts = [mock_part]
        sys.modules['eval.danielson'].generate_ai_content.return_value = mock_response
        sys.modules['eval.danielson'].normalize_score_integer.return_value = 3
        
        # Create model
        model = DanielsonModel()
        
        # Call generate_component_evaluation
        result = model.generate_component_evaluation(
            component_id="1a",
            observation_text="Sample observation",
            context="Sample context"
        )
        
        # Verify the function was called
        sys.modules['eval.danielson'].generate_ai_content.assert_called_once()
        
        # Verify the result
        assert result["score"] == 3
        assert result["summary"] == "Sample evaluation summary"
        
        # Test error handling
        sys.modules['eval.danielson'].generate_ai_content.side_effect = Exception("Test error")
        result = model.generate_component_evaluation("1a", "Sample observation", "Sample context")
        assert result["score"] == 1
        assert "Error" in result["summary"]

    def test_restructure_component_feedback(self):
        """Test restructuring component feedback."""
        # Setup mock
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "Restructured feedback"
        mock_response.parts = [mock_part]
        sys.modules['eval.danielson'].generate_ai_content.return_value = mock_response
        
        # Create model
        model = DanielsonModel()
        
        # Call restructure_component_feedback
        result = model.restructure_component_feedback(
            text="Original feedback",
            evidence="Sample evidence",
            component_id="1a"
        )
        
        # Verify the function was called
        sys.modules['eval.danielson'].generate_ai_content.assert_called_once()
        
        # Verify the result
        assert result == "Restructured feedback"
        
        # Test error handling
        sys.modules['eval.danielson'].generate_ai_content.side_effect = Exception("Test error")
        result = model.restructure_component_feedback("Original feedback", "Sample evidence", "1a")
        assert result == "Original feedback"  # Should return original text on error

    def test_generate_single_component_evaluation(self):
        """Test generating a single component evaluation."""
        # Reset mocks
        sys.modules['eval.danielson'].generate_ai_content.reset_mock()
        sys.modules['eval.danielson'].generate_ai_content.side_effect = None
        
        # Setup mocks for internal methods
        with patch.object(DanielsonModel, 'analyze_danielson_context') as mock_analyze, \
             patch.object(DanielsonModel, 'generate_component_evaluation') as mock_generate_eval, \
             patch.object(DanielsonModel, 'restructure_component_feedback') as mock_restructure:
            
            mock_analyze.return_value = {"analysis": "Context analysis", "error": None}
            mock_generate_eval.return_value = {"score": 3, "summary": "Evaluation summary"}
            mock_restructure.return_value = "Enhanced feedback"
            sys.modules['eval.danielson'].normalize_score_integer.return_value = 3
            
            # Create model
            model = DanielsonModel()
            
            # Call generate_single_component_evaluation
            result = model.generate_single_component_evaluation(
                low_inference_notes="Sample observation",
                component_id="1a"
            )
            
            # Verify the function calls
            mock_analyze.assert_called_once_with("Sample observation", model)
            mock_generate_eval.assert_called_once_with(
                component_id="1a",
                observation_text="Sample observation",
                context="Context analysis",
                model=model
            )
            mock_restructure.assert_called_once_with(
                text="Evaluation summary",
                evidence="Sample observation",
                component_id="1a",
                model=model
            )
            
            # Verify the result
            assert result["component_id"] == "1a"
            assert result["score"] == 3
            assert result["summary"] == "Enhanced feedback"
            assert result["domain"] == "1"
            
            # Test invalid component ID
            result = model.generate_single_component_evaluation("Sample", "invalid")
            assert "error" in result
            assert "Invalid component ID" in result["error"]
            
            # Test context analysis error
            mock_analyze.return_value = {"analysis": "", "error": "Context error"}
            result = model.generate_single_component_evaluation("Sample", "1a")
            assert "error" in result
            assert result["error"] == "Context error"

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 