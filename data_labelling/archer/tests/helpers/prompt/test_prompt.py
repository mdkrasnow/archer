import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from helpers.prompt import Prompt

class TestPrompt:
    def test_prompt_initialization(self):
        """Test that a Prompt object initializes with correct default values"""
        content = "This is a test prompt"
        prompt = Prompt(content)
        
        assert prompt.content == content
        assert prompt.score == 0.0
        assert prompt.generation == 0
        assert prompt.feedback == ""
        assert prompt.history == []
        assert hasattr(prompt, "llm_call")
    
    def test_prompt_initialization_with_custom_values(self):
        """Test that a Prompt object initializes with custom values"""
        content = "This is a test prompt"
        score = 4.5
        generation = 2
        
        prompt = Prompt(content, score, generation)
        
        assert prompt.content == content
        assert prompt.score == score
        assert prompt.generation == generation
        assert prompt.feedback == ""
        assert prompt.history == []
    
    def test_prompt_update(self):
        """Test that a Prompt can be updated and history is tracked correctly"""
        original_content = "Original prompt"
        original_score = 0.0
        prompt = Prompt(original_content)
        
        new_content = "Updated prompt"
        new_score = 4.5
        feedback = "This prompt is better"
        
        prompt.update(new_content, new_score, feedback)
        
        # Check that the prompt was updated
        assert prompt.content == new_content
        assert prompt.score == new_score
        assert prompt.feedback == feedback
        assert prompt.generation == 1
        
        # Check that history was tracked correctly
        assert len(prompt.history) == 1
        assert prompt.history[0] == (original_content, original_score, "")
    
    def test_multiple_updates(self):
        """Test that multiple updates are tracked correctly in history"""
        prompt = Prompt("First version")
        
        # First update
        prompt.update("Second version", 3.0, "Getting better")
        
        # Second update
        prompt.update("Third version", 4.0, "Even better")
        
        # Third update
        prompt.update("Final version", 4.8, "Nearly perfect")
        
        # Check current state
        assert prompt.content == "Final version"
        assert prompt.score == 4.8
        assert prompt.feedback == "Nearly perfect"
        assert prompt.generation == 3
        
        # Check history
        assert len(prompt.history) == 3
        assert prompt.history[0] == ("First version", 0.0, "")
        assert prompt.history[1] == ("Second version", 3.0, "Getting better")
        assert prompt.history[2] == ("Third version", 4.0, "Even better")
    
    def test_string_representation(self):
        """Test the string representation of a Prompt"""
        content = "This is a very long prompt that should be truncated in the string representation"
        prompt = Prompt(content, 4.2, 3)
        
        string_repr = str(prompt)
        
        assert "Gen 3" in string_repr
        assert "Score: 4.20" in string_repr
        assert content[:50] in string_repr
        assert "..." in string_repr

    def test_prompt_with_empty_content(self):
        """Test behavior with empty prompt content"""
        prompt = Prompt("")
        
        assert prompt.content == ""
        assert str(prompt) == "Prompt (Gen 0, Score: 0.00): ..."
    
    def test_prompt_with_very_long_content(self):
        """Test with an extremely long prompt content"""
        long_content = "a" * 10000
        prompt = Prompt(long_content)
        
        assert prompt.content == long_content
        assert len(str(prompt)) < len(long_content)
        
    def test_prompt_with_special_characters(self):
        """Test with special characters in the prompt content"""
        content_with_special_chars = "!@#$%^&*()_+<>?:{}\n\t"
        prompt = Prompt(content_with_special_chars)
        
        assert prompt.content == content_with_special_chars 