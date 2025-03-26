from typing import List, Dict, Tuple, Any, Optional
import random
from helpers.llm_call import llm_call

class Prompt:
    """
    Represents a prompt being evaluated and optimized in the system.
    
    A prompt is the instruction given to the generative model that guides its output.
    Prompts evolve over time based on feedback and performance.
    """
    
    def __init__(self, content: str, score: float = 0.0, generation: int = 0):
        """
        Initialize a new Prompt.
        
        Args:
            content: The actual text of the prompt
            score: The performance score of this prompt (default: 0.0)
            generation: Which evolutionary generation this prompt belongs to (default: 0)
        """
        # The actual text content of the prompt
        self.content = content
        
        # Performance score from evaluations (higher is better)
        self.score = score
        
        # Which generation this prompt belongs to in the evolutionary process
        self.generation = generation
        
        # Feedback received about this prompt's performance
        self.feedback = ""
        
        # History of all previous versions of this prompt
        # Stored as tuples of (content, score, feedback)
        self.history = []

        self.llm_call = llm_call
    
    def update(self, new_content: str, score: float, feedback: str):
        """
        Update prompt with new content and record history.
        
        This method preserves the evolutionary history by storing the previous
        version before updating to the new content.
        
        Args:
            new_content: The new prompt text
            score: The new performance score
            feedback: Feedback explaining the performance
        """
        # Save current state to history before updating
        self.history.append((self.content, self.score, self.feedback))
        
        # Update to new state
        self.content = new_content
        self.score = score
        self.feedback = feedback
        
        # Increment generation counter
        self.generation += 1
        
    def __str__(self):
        """
        Return a string representation of the prompt.
        
        Returns:
            A string containing the generation, score, and preview of the content
        """
        return f"Prompt (Gen {self.generation}, Score: {self.score:.2f}): {self.content[:50]}..."
