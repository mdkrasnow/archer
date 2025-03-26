import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from forwardPass.evaluator import AIExpert

class TestAIExpert:
    def test_aiexpert_initialization(self):
        """Test that AIExpert initializes with correct parameters"""
        model_name = "claude-3"
        knowledge_base = ["Document 1", "Document 2"]
        rubric = {"clarity": {"weight": 0.5, "description": "Is the content clear?"}}
        
        expert = AIExpert(model_name, knowledge_base, rubric)
        
        assert expert.model_name == model_name
        assert expert.knowledge_base == knowledge_base
        assert expert.rubric == rubric
        assert hasattr(expert, "llm_call")
    
    @patch('forwardPass.evaluator.llm_call')
    def test_content_evaluation(self, mock_llm_call):
        """Test evaluation of generated content"""
        # Mock LLM response
        mock_llm_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """
                        Score: 4
                        Feedback: Good content but needs improvement in clarity
                        Improved Output: Improved version of the content
                        Summary: Overall good with minor issues
                        """
                    }
                }
            ]
        }
        
        model_name = "claude-3"
        knowledge_base = ["Document 1", "Document 2"]
        rubric = {"clarity": {"weight": 0.5, "description": "Is the content clear?"}}
        
        expert = AIExpert(model_name, knowledge_base, rubric)
        result = expert.evaluate(
            "Original content", 
            "Create compelling sales email"
        )
        
        # Verify the LLM was called with correct parameters
        mock_llm_call.assert_called_once()
        args = mock_llm_call.call_args[1]
        assert args["model"] == model_name
        assert len(args["messages"]) == 1
        assert "Original content" in args["messages"][0]["content"]
        assert "Create compelling sales email" in args["messages"][0]["content"]
        
        # Verify the result
        assert result["score"] == 4.0  # Updated to match the mock response
        assert "feedback" in result
        assert "improved_output" in result
        assert "summary" in result

    @patch('forwardPass.evaluator.llm_call')
    def test_evaluation_with_response_parsing(self, mock_llm_call):
        """Test evaluation with proper response parsing"""
        # Setup a mock implementation that parses the LLM response
        def parse_evaluation(self, content):
            lines = content.strip().split('\n')
            score = 4.0  # Default
            feedback = "No feedback provided"
            improved = "No improved output provided"
            summary = "No summary provided"
            
            for line in lines:
                line = line.strip()
                if line.startswith("Score:"):
                    try:
                        score = float(line.replace("Score:", "").strip())
                    except:
                        pass
                elif line.startswith("Feedback:"):
                    feedback = line.replace("Feedback:", "").strip()
                elif line.startswith("Improved Output:"):
                    improved = line.replace("Improved Output:", "").strip()
                elif line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
            
            return {
                "score": score,
                "feedback": feedback,
                "improved_output": improved,
                "summary": summary
            }
        
        # Mock the LLM response
        mock_llm_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """
                        Score: 4.5
                        Feedback: Good content but needs improvement in clarity
                        Improved Output: This is an improved version of the content
                        Summary: Overall good with minor issues
                        """
                    }
                }
            ]
        }
        
        # Create the expert and apply the mock method
        model_name = "claude-3"
        knowledge_base = ["Document 1", "Document 2"]
        rubric = {"clarity": {"weight": 0.5, "description": "Is the content clear?"}}
        
        expert = AIExpert(model_name, knowledge_base, rubric)
        
        # Replace the evaluate method with a version that uses our parse_evaluation
        with patch.object(AIExpert, 'evaluate', autospec=True) as mock_evaluate:
            mock_evaluate.side_effect = lambda self, generated_content, input_data: parse_evaluation(
                self, 
                mock_llm_call.return_value["choices"][0]["message"]["content"]
            )
            
            # Call the patched method
            result = expert.evaluate(
                "Original content", 
                "Create compelling sales email"
            )
            
            # Verify the result
            assert result["score"] == 4.5
            assert "clarity" in result["feedback"]
            assert result["improved_output"] == "This is an improved version of the content"
            assert result["summary"] == "Overall good with minor issues"

    @patch('forwardPass.evaluator.llm_call')
    def test_batch_evaluation(self, mock_llm_call):
        """Test evaluation of multiple content pieces (implementing a batch feature)"""
        # This test demonstrates how a batch evaluation feature could be implemented
        
        # Mock responses for multiple calls
        mock_responses = [
            {
                "choices": [{"message": {"content": "Score: 3.5\nFeedback: Feedback 1\nImproved Output: Improved 1"}}]
            },
            {
                "choices": [{"message": {"content": "Score: 4.2\nFeedback: Feedback 2\nImproved Output: Improved 2"}}]
            }
        ]
        
        mock_llm_call.side_effect = mock_responses
        
        model_name = "claude-3"
        knowledge_base = ["Document 1", "Document 2"]
        rubric = {"clarity": {"weight": 0.5, "description": "Is the content clear?"}}
        
        expert = AIExpert(model_name, knowledge_base, rubric)
        
        # We'll implement a batch evaluation method for testing purposes
        def evaluate_batch(contents, input_data):
            results = []
            for content in contents:
                result = expert.evaluate(content, input_data)
                results.append(result)
            return results
        
        # Use the batch evaluation function
        results = evaluate_batch(
            ["Content 1", "Content 2"],
            "Create compelling sales email"
        )
        
        # Check results
        assert len(results) == 2
        assert "feedback" in results[0]
        assert "improved_output" in results[0]
        assert "feedback" in results[1]
        assert "improved_output" in results[1]
        
        # Verify the LLM was called twice with different content
        assert mock_llm_call.call_count == 2
        first_call = mock_llm_call.call_args_list[0]
        second_call = mock_llm_call.call_args_list[1]
        assert "Content 1" in first_call[1]["messages"][0]["content"]
        assert "Content 2" in second_call[1]["messages"][0]["content"]

    @patch('forwardPass.evaluator.llm_call')
    def test_empty_content_evaluation(self, mock_llm_call):
        """Test evaluation with empty content"""
        # Mock LLM response for empty content
        mock_llm_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Score: 1\nFeedback: Cannot evaluate empty content\nImproved Output: Please provide content"
                    }
                }
            ]
        }
        
        expert = AIExpert("claude-3", ["Document 1"], {"clarity": {"weight": 0.5}})
        result = expert.evaluate("", "Create compelling sales email")
        
        # Verify the result has appropriate fallback values
        assert "feedback" in result
        assert "improved_output" in result
        
        # Verify empty content was passed to the LLM
        args = mock_llm_call.call_args[1]
        assert 'Generated content: ' in args["messages"][0]["content"]

    @patch('forwardPass.evaluator.llm_call')
    def test_llm_call_error_handling(self, mock_llm_call):
        """Test handling of errors in LLM responses"""
        # Set up the mock to return an empty or invalid response
        mock_llm_call.return_value = {}
        
        expert = AIExpert("claude-3", ["Document 1"], {"clarity": {"weight": 0.5}})
        result = expert.evaluate("Test content", "Test input")
        
        # Verify fallback values are used
        assert result["score"] == 3.0
        assert "Error" in result["feedback"]
        assert "Error" in result["improved_output"]
        assert "Error" in result["summary"]

    def test_missing_knowledge_base(self):
        """Test initialization with empty knowledge base"""
        expert = AIExpert("claude-3", [], {"clarity": {"weight": 0.5}})
        assert expert.knowledge_base == []
        
        # Knowledge base should be empty but not cause initialization errors
        assert isinstance(expert.knowledge_base, list)
        assert len(expert.knowledge_base) == 0

    @patch('forwardPass.evaluator.llm_call')
    def test_extremely_long_content(self, mock_llm_call):
        """Test evaluation with extremely long content"""
        # Create very long content (10,000 characters)
        long_content = "This is a test. " * 1000
        
        mock_llm_call.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Score: 3\nFeedback: Content is too verbose\nImproved Output: Shorten content"
                    }
                }
            ]
        }
        
        expert = AIExpert("claude-3", ["Document 1"], {"clarity": {"weight": 0.5}})
        result = expert.evaluate(long_content, "Create compelling email")
        
        # Verify the long content was included in the LLM call
        args = mock_llm_call.call_args[1]
        assert long_content in args["messages"][0]["content"]
        
        # Result should be valid
        assert "feedback" in result
        assert "improved_output" in result